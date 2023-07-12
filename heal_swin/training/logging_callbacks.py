import os
import sys
import socket
import subprocess
import signal
import re
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import shutil

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback, GPUStatsMonitor
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities import rank_zero_only
import torchmetrics

from heal_swin.utils import get_paths


class MyMLFlowLogger(MLFlowLogger):
    """Simple wrapper around MLFlowLogger"""

    def __init__(self, step_offset=0, **kwargs):
        super().__init__(**kwargs)
        self.step_offset = step_offset

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        step_shifted = step + self.step_offset
        super().log_metrics(metrics, step_shifted)


class MLFlowLogging(Callback):
    def __init__(self, job_id, mlflow_tags, mlflow_params, log_dir, train_samples):
        super().__init__()
        self.mlflow_tags = mlflow_tags
        self.mlflow_params = mlflow_params
        self.job_id = job_id
        self.set_run_name()
        self.log_dir = log_dir
        self.train_samples = train_samples
        self.setup_done = False
        self.killed = False

    def count_parameters(self, model):
        total_param = 0
        output_str = ""
        for name, param in model.named_parameters():
            if param.requires_grad:
                num_param = np.prod(param.size())
                if param.dim() > 1:
                    decomp_str = "x".join(str(x) for x in list(param.size()))
                    output_str += name + ": " + decomp_str + " = " + str(num_param) + "\n"
                else:
                    output_str += name + ": " + str(num_param) + "\n"
                total_param += num_param
        return total_param, output_str

    def log_profiler_results(self):
        metrics = self.mlf_client.get_run(self.run_id).data.metrics.keys()
        metric_name = "train_time_per_sample in ms"
        if metric_name in metrics:
            return
        if not os.path.isdir(self.artifact_path):
            return
        for name in os.listdir(self.artifact_path):
            m = re.fullmatch(r"fit-profiling-results-?([0-9]*)\.txt", name)
            if m is not None and m.group(1) in ["", "0"]:
                with open(Path(self.artifact_path) / name, "r") as f:
                    profiler_results = f.read()
                m = re.search(r"run_training_epoch\s*\|\s*([0-9\.]*)[^0-9]", profiler_results)
                epoch_time = float(m.group(1))
                sample_time = epoch_time * 1000 / self.train_samples
                self.log_metric(metric_name, sample_time)

    def copy_log_dir_to_artifacts(self):
        if self.job_id is not None:
            slurm_path = os.path.join(get_paths.get_slurm_path(), "slurm-" + self.job_id + ".out")
            dst_path = os.path.join(self.log_dir.name, "slurm-" + self.job_id + ".txt")
            if os.path.isfile(slurm_path):
                shutil.copyfile(slurm_path, dst_path)

        os.makedirs(self.artifact_path, exist_ok=True)

        command = ["rsync", "-rq", "--delete"]
        command += [os.path.join(self.log_dir.name, ""), self.artifact_path]
        subprocess.run(command)

        self.log_profiler_results()

        return self.artifact_path

    def set_run_name(self):
        if self.job_id is not None:
            self.run_name = self.job_id
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            self.run_name = timestamp + "_" + socket.gethostname()

    def log_param(self, name, value):
        self.mlf_client.log_param(self.run_id, name, value)

    def log_tag(self, name, value):
        self.mlf_client.set_tag(self.run_id, name, value)

    def log_metric(self, name, value):
        self.mlf_client.log_metric(self.run_id, name, value)

    def kill_run(self, reason=None, handler=None):
        if not self.killed:
            suffix = "logging run as killed in MLFlow..."
            if reason == "exception":
                print("Exception detected, " + suffix)
            elif reason == signal.SIGTERM:
                print("SIGTERM detected, " + suffix)
            elif reason == "Keyboard interrupt":
                print("Keyboard interrupt detected, " + suffix)
            else:
                print(f"kill_run was called with reason={reason}, " + suffix)
            self.pl_module.logger.finalize(status="KILLED")
            self.copy_log_dir_to_artifacts()
            self.log_dir.cleanup()

        self.killed = True

        if reason != "exception":  # exceptions are raised after kill_run is called, for debugging
            sys.exit(1)

    def setup(self, trainer, pl_module, stage):
        if self.setup_done:
            return
        self.deactivate = not trainer.is_global_zero  # only log on global rank zero instance
        if self.deactivate:
            return
        self.pl_module = pl_module
        self.run_id = pl_module.logger.run_id
        self.mlf_client = pl_module.logger.experiment

        expmt_name = self.mlf_client.get_experiment(pl_module.logger.name).name
        print(f"This run has id {self.run_id}")
        print(f"It is saved in MLFlow experiment {expmt_name} with nr {pl_module.logger.name}")

        total_params, param_decomp = self.count_parameters(pl_module)
        self.mlflow_params["total_parameters"] = str(total_params)

        if torch.cuda.is_available():
            devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
            device_names = [torch.cuda.get_device_name(device) for device in devices]
            self.mlflow_params["devices"] = " + ".join(device_names)
        else:
            self.mlflow_params["devices"] = "cpu"

        self.mlflow_params.update(pl_module.mlflow_params)
        for key, value in self.mlflow_params.items():
            self.log_param(key, value)

        self.mlflow_tags.update(pl_module.mlflow_tags)
        self.mlflow_tags["mlflow.runName"] = self.run_name
        for key, value in self.mlflow_tags.items():
            self.log_tag(key, value)

        artifact_uri = self.mlf_client.get_run(self.run_id).info.artifact_uri
        self.artifact_path = artifact_uri.replace("file://", "")

        self.copy_log_dir_to_artifacts()

        if hasattr(self.pl_module, "additional_parameters"):
            for key, val in self.pl_module.additional_parameters.items():
                self.log_param(key, val)

        if hasattr(self.pl_module, "files_to_save"):
            files = self.pl_module.files_to_save

            if isinstance(files, str):
                files = [files]

            print()
            for file_to_save in files:
                if os.path.isfile(file_to_save) and os.path.isdir(self.log_dir.name):
                    print(
                        f"Copying {file_to_save} to {self.artifact_path} ({self.log_dir.name})"
                    )  # Technically not true, but that's where you'll find it after the run is over
                    shutil.copy(file_to_save, self.log_dir.name)
            print()

        self.setup_done = True

    def on_fit_end(self, trainer, pl_module):
        if self.deactivate:
            return
        self.copy_log_dir_to_artifacts()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.deactivate:
            return
        checkpoint["run_id"] = self.run_id
        checkpoint["mlflow_expmt"] = pl_module.logger.name
        return checkpoint

    def on_train_epoch_start(self, trainer, pl_module):
        if self.deactivate:
            return
        self.copy_log_dir_to_artifacts()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.deactivate:
            return
        self.copy_log_dir_to_artifacts()

    def teardown(self, trainer, pl_module, stage):
        if self.deactivate:
            return
        self.copy_log_dir_to_artifacts()


class MLFlowGPUStatsMonitor(GPUStatsMonitor):
    """Format metric names correctly for characters allowed by mlflow"""

    @staticmethod
    def _parse_gpu_stats(
        gpu_ids: str, stats: List[List[float]], keys: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        logs = super(MLFlowGPUStatsMonitor, MLFlowGPUStatsMonitor)._parse_gpu_stats(
            gpu_ids=gpu_ids, stats=stats, keys=keys
        )
        logs = {k.replace(":", ""): v for k, v in logs.items()}
        logs = {k.replace("(MB)", "in MB"): v for k, v in logs.items()}
        logs = {k.replace("(°C)", "in degree C"): v for k, v in logs.items()}
        logs = {k.replace("(%)", "in percent"): v for k, v in logs.items()}
        return logs


class ValMaskedIoULogger(Callback):
    def __init__(self, prefix, f_out):
        if prefix != "" and prefix[-1] != "_":
            prefix = prefix + "_"
        self.prefix = prefix

        self.iou = torchmetrics.IoU(num_classes=f_out, reduction="none")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.iou.to(pl_module.device)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        img, mask = batch
        outputs = outputs[mask != 0]
        mask = mask[mask != 0]

        self.iou.update(outputs, mask.to(pl_module.device))

    def on_validation_epoch_end(self, trainer, pl_module):
        class_ious = self.iou.compute().cpu()
        m_iou = torch.mean(class_ious)
        m_iou_ignored = torch.mean(class_ious[1:])
        metrics = {
            f"{self.prefix}val_masked_iou": m_iou.item(),
            f"{self.prefix}val_masked_iou_ignored": m_iou_ignored.item(),
        }

        trainer.logger.log_metrics(metrics)
