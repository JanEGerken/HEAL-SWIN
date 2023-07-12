#!usr/bin/env python3
import argparse
import tempfile
import os
import sys
import shutil
from pathlib import Path
from dataclasses import asdict

import matplotlib.pyplot as plt

import pytorch_lightning as pl

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)

from heal_swin.utils import get_paths, serialize, utils
from heal_swin.training import logging_callbacks as callbacks
from heal_swin.models_lightning import models_lightning

from heal_swin import evaluate
from heal_swin.evaluation import evaluate_config

from heal_swin.data.data import get_data_module

from heal_swin.utils.mlflow_utils import get_tracking_uri


def get_effective_batch_size(pl_config, config):
    batch_size = config.common.batch_size
    batch_size *= pl_config.num_nodes
    batch_size *= 1 if pl_config.gpus == 0 or pl_config.gpus is None else pl_config.gpus
    batch_size *= (
        1 if pl_config.accumulate_grad_batches is None else pl_config.accumulate_grad_batches
    )
    return batch_size


def get_train_samples(pl_config, config, dm):
    batch_size = get_effective_batch_size(pl_config, config)
    if pl_config.fast_dev_run is True:
        return batch_size
    elif type(pl_config.fast_dev_run) is int:
        return pl_config.fast_dev_run * batch_size
    elif type(pl_config.limit_train_batches) is int:
        return pl_config.limit_train_batches * batch_size
    elif type(pl_config.limit_train_batches) is float and pl_config.limit_train_batches < 1.0:
        return int(pl_config.limit_train_batches * len(dm.train_dataset))
    elif type(pl_config.overfit_batches) is int:
        return pl_config.overfit_batches * batch_size
    elif type(pl_config.overfit_batches) is float and pl_config.overfit_batches != 0.0:
        return int(pl_config.overfit_batches * len(dm.train_dataset))
    else:
        return len(dm.train_dataset)


def get_mlflow_params_tags(pl_config, data_config, train_config, dm):
    mlflow_params = {
        "len_train_data": len(dm.train_dataset),
        "len_val_data": len(dm.val_dataset),
        "len_pred_data": len(dm.pred_dataset),
        "effective_train_batch_size": get_effective_batch_size(pl_config, data_config),
        "train_samples": get_train_samples(pl_config, data_config, dm),
    }
    mlflow_tags = {"cmd": " ".join(sys.argv)}
    if data_config.common.manual_overfit_batches > 0:
        mlflow_tags["overfit_imgs"] = "\n".join(dm.get_train_overfit_names())

    if train_config.description is not None:
        mlflow_tags["mlflow.note.content"] = train_config.description

    return mlflow_params, mlflow_tags


def get_callbacks(dm, train_config, pl_config, log_dir, data_config):
    cbs = {}

    checkpoint_cb = ModelCheckpoint(
        monitor=train_config.ckpt_metric,
        mode=train_config.ckpt_mode,
        dirpath=log_dir.name,
        filename="{epoch}_{" + train_config.ckpt_metric + ":.4f}",
        save_top_k=3,
        save_last=True,
    )
    cbs["checkpoint"] = checkpoint_cb

    mlflow_params, mlflow_tags = get_mlflow_params_tags(pl_config, data_config, train_config, dm)
    mlf_logging_cb = callbacks.MLFlowLogging(
        train_config.job_id,
        mlflow_tags,
        mlflow_params,
        log_dir,
        mlflow_params["train_samples"],
    )
    cbs["mlf_logging"] = mlf_logging_cb

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    cbs["lr_monitor"] = lr_monitor

    early_stopping = EarlyStopping(
        monitor=train_config.early_stopping_monitor,
        min_delta=train_config.early_stopping_min_delta,
        patience=train_config.early_stopping_patience,
        mode=train_config.early_stopping_mode,
        verbose=True,
    )
    if train_config.early_stopping:
        cbs["early_stopping"] = early_stopping

    if train_config.log_gpu_stats and pl_config.gpus is not None and pl_config.gpus > 0:
        cbs["gpu_stats"] = callbacks.MLFlowGPUStatsMonitor()

    return cbs


def save_config(
    output_path,
    config_path,
    train_config,
    pl_config,
    model_config,
    data_spec,
    data_config,
    run_config,
):
    output_path = Path(output_path)
    serialize.save(train_config, output_path / "train_config")
    serialize.save(pl_config, output_path / "pl_config")
    serialize.save(model_config, output_path / "model_config")
    serialize.save(data_spec, output_path / "data_spec")
    serialize.save(data_config, output_path / "data_config")
    serialize.save(run_config, output_path / "run_config")

    abs_config_path = get_paths.get_abs_path_from_config_path(config_path)
    config_name = os.path.basename(abs_config_path)

    if config_name == "slurm_script":
        config_name = f"train_config_{train_config.job_id}.py"

    shutil.copyfile(abs_config_path, output_path / config_name)


def train_model(
    config_path, train_config, pl_config, model_class, model_config, data_config, run_config
):
    if pl_config.overfit_batches > 0:
        print("overfit_batches is set. Please use manual_overfit_batches instead")
        sys.exit(1)

    train_config.seed = pl.utilities.seed.seed_everything(train_config.seed, workers=True)

    dm, data_spec = get_data_module(data_config)

    mlf_logger = callbacks.MyMLFlowLogger(
        step_offset=train_config.logging_step_offset,
        experiment_name=train_config.mlflow_expmt,
        tracking_uri=get_tracking_uri(),
    )

    log_dir = tempfile.TemporaryDirectory()
    save_config(
        log_dir.name,
        config_path=config_path,
        train_config=train_config,
        pl_config=pl_config,
        model_config=model_config,
        data_config=data_config,
        data_spec=data_spec,
        run_config=run_config,
    )

    profiler = SimpleProfiler(dirpath=log_dir.name, filename="profiling-results")

    cbs = get_callbacks(dm, train_config, pl_config, log_dir, data_config)

    trainer = pl.Trainer(
        **asdict(pl_config),
        logger=mlf_logger,
        profiler=profiler,
        callbacks=list(cbs.values()),
        plugins=[DDPPlugin(find_unused_parameters=False)],
        weights_summary="full",
    )

    try:

        if train_config.load_checkpoint is not None:
            assert os.path.isfile(train_config.load_checkpoint)
            model = model_class.load_from_checkpoint(
                train_config.load_checkpoint,
                config=model_config,
                data_spec=data_spec,
                data_config=data_config,
            )
        else:
            model = model_class(config=model_config, data_spec=data_spec, data_config=data_config)

        if pl_config.auto_lr_find:
            res = trainer.tune(model, dm.train_dataloader(), dm.val_dataloader())

            lr_finder = res["lr_find"]

            lr_finder.results

            # Plot scan
            fig = lr_finder.plot(suggest=True)  # noqa: F841
            plt.savefig(str(Path(log_dir.name) / "lr_plot.png"))

            # Pick point based on plot, or get suggestion
            model.config.optimizer_config.learning_rate = lr_finder.suggestion()
            model.learning_rate = model.config.optimizer_config.learning_rate

        mlf_logger.log_hyperparams(serialize.dataclass_to_normalized_json(train_config, "train"))
        mlf_logger.log_hyperparams(serialize.dataclass_to_normalized_json(model_config, "model"))
        mlf_logger.log_hyperparams(serialize.dataclass_to_normalized_json(data_config, "data"))
        mlf_logger.log_hyperparams(serialize.dataclass_to_normalized_json(data_spec, "data_spec"))
        mlf_logger.log_hyperparams(dict(model_class=model_class.__name__))
        mlf_logger.log_hyperparams(dict(dataset=data_config.__class__.__name__))
        mlf_logger.log_hyperparams(
            {f"args_{key}": value for key, value in asdict(pl_config).items()}
        )

        trainer.fit(model, dm)

        if trainer.is_global_zero:
            best_ckpt_src_path = cbs["checkpoint"].best_model_path
            cbs["mlf_logging"].log_param("best_checkpoint", os.path.basename(best_ckpt_src_path))
            best_ckpt_dst_path = Path(log_dir.name) / "best.ckpt"
            shutil.copy(best_ckpt_src_path, best_ckpt_dst_path)

        if not cbs["mlf_logging"].deactivate:
            artifacts_path = cbs["mlf_logging"].copy_log_dir_to_artifacts()

            print(f"\nSaved artifacts to {artifacts_path}")

        model.logger.finalize(status="FINISHED")
    except:  # noqa:722
        cbs["mlf_logging"].kill_run("exception")
        log_dir.cleanup()
        raise

    log_dir.cleanup()

    do_eval = trainer.is_global_zero and train_config.eval_after_train

    del trainer
    del model
    del dm

    if do_eval:
        eval_config = evaluate_config.EvaluateConfig(
            path=mlf_logger._run_id,
            eval_config_name="end_of_train_eval_config",
            epoch="best",
            metric_prefix="best",
            pred_writer=None,
            validate=True,
            predict=True,
            train_config=train_config,
            data_config=data_config,
        )

        evaluate.evaluate(eval_config, pl_config, config_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="run_configs/default_train_run_config.py",
        help="path to config python file, absolute or relative to HEAL-SWIN path",
    )

    args = parser.parse_args()

    train_config = utils.get_config_from_config_path(args.config_path, "get_train_run_config")
    pl_config = utils.get_config_from_config_path(args.config_path, "get_pl_config")

    train_model(
        config_path=args.config_path,
        train_config=train_config.train,
        pl_config=pl_config,
        model_class=models_lightning.MODEL_FROM_CONFIG_NAME[train_config.model.__class__.__name__],
        model_config=train_config.model,
        data_config=train_config.data,
        run_config=train_config,
    )


if __name__ == "__main__":
    main()
