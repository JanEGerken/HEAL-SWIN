import os
import re
from pathlib import Path

import mlflow

from heal_swin.utils import mlflow_utils


class ValidateMlflowRun:
    def __init__(self, slurm_path):
        if os.path.isfile(slurm_path):
            with open(slurm_path, "r") as f:
                slurm_content = f.read()
            m = re.search("This run has id (.*)", slurm_content)
            if m is not None:
                self.run_id = m.groups()[0]
        self.metrics = []
        self.tags = ["cmd"]
        self.params = ["job_id", "mlflow_expmt"]
        self.files = []  # these can be regexps
        mlflow.set_tracking_uri(mlflow_utils.get_tracking_uri())

    def validate_mlflow_run(self, run_id=None):
        if run_id is None and self.run_id is not None:
            run_id = self.run_id
        else:
            assert 0 == 1, "No run id found"

        self.run = mlflow.get_run(run_id)
        self.artifacts_path = self.run.info.artifact_uri.replace("file://", "")
        self.file_list = []
        for name in os.listdir(self.artifacts_path):
            if os.path.isfile(Path(self.artifacts_path) / name):
                self.file_list.append(name)

        assert self.run.info.status == "FINISHED"
        assert os.path.isdir(self.artifacts_path)

        for metric in self.metrics:
            assert metric in self.run.data.metrics, f"metric {metric} not found"
        for param in self.params:
            matching_params = [
                run_param for run_param in self.run.data.params if param in run_param
            ]
            assert len(matching_params), f"parameter {param} not found"
        for tag in self.tags:
            assert tag in self.run.data.tags, f"tag {tag} not found"
        for name in self.files:
            assert any(re.fullmatch(name, f) for f in self.file_list), f"file {name} not found"

        return True


class ValidateMlflowTrainRun(ValidateMlflowRun):
    def __init__(self, slurm_path):
        super().__init__(slurm_path)
        self.metrics += [
            "epoch",
            "train_acc",
            "train_acc_ignored",
            "train_iou_global",
            "train_iou_global_ignored",
            "val_acc",
            "val_acc_ignored",
            "val_iou_global",
            "val_iou_global_ignored",
            "val_loss",
        ]
        self.params += [
            "batch_size",
            "cam_pos",
            "ckpt_metric",
            "ckpt_mode",
            "crop_green",
            "dataset",
            "devices",
            "early_stopping",
            "early_stopping_min_delta",
            "early_stopping_mode",
            "early_stopping_monitor",
            "early_stopping_patience",
            "f_in",
            "f_out",
            "len_train_data",
            "len_val_data",
            "load_checkpoint",
            "optimizer_config.learning_rate",
            "seed",
            "shuffle",
            "total_parameters",
            "train_share",
            "train_worker",
            "val_worker",
            "val_batch_size",
        ]
        self.files += [
            "pl_config",
            "data_spec",
            "model_config",
            "train_config",
            "last.ckpt",
            r"epoch=[0-9]+_[^0-9]+=[0-9.]+\.ckpt",
        ]
