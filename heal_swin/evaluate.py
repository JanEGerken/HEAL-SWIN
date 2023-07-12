import os
import argparse
from dataclasses import asdict
import shutil

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

import heal_swin.training.logging_callbacks as callbacks
from heal_swin.models_lightning import models_lightning
from heal_swin.utils import serialize
from heal_swin.data import data

from heal_swin.evaluation.evaluate_config import EvaluateConfig
from heal_swin.training.train_config import PLConfig
from heal_swin.utils import get_paths, utils
from heal_swin.utils.mlflow_utils import get_tracking_uri


def evaluate(eval_config: EvaluateConfig, pl_config: PLConfig, config_path: str):
    """Does evaluation from a given EvaluateConfig object.
    All information that can be derived from the run
    identifier is extracted and used here."""

    ckpt_path, artifact_path, explicit_path = utils.check_and_get_ckpt_paths(
        run_identifier=eval_config.path,
        epoch=eval_config.epoch,
        epoch_number=eval_config.epoch_number,
    )
    ckpt_name = ckpt_path.split("/")[-1]
    serialize.save(eval_config, os.path.join(artifact_path, eval_config.eval_config_name))

    abs_config_path = get_paths.get_abs_path_from_config_path(config_path)
    config_name = os.path.basename(abs_config_path)
    if config_name == "slurm_script":
        config_name = f"eval_config_{eval_config.eval_config_name}.py"

    shutil.copyfile(abs_config_path, os.path.join(artifact_path, config_name))

    model_config = serialize.load(os.path.join(artifact_path, "model_config"))

    run_id = artifact_path.split("/")[-2]

    datamodule, data_spec = data.get_data_module(eval_config.data_config)

    if eval_config.metric_prefix is None:
        if eval_config.epoch.lower() == "number":
            epoch = "epoch=" + str(eval_config.epoch_number)
        elif explicit_path:
            epoch = ckpt_name
        else:
            epoch = eval_config.epoch  # in this case eval_config is either "best" or "last"

        metric_prefix = "evaluate_" + epoch + "_"
    else:
        metric_prefix = eval_config.metric_prefix + "_"

    model_cls = models_lightning.MODEL_FROM_NAME[
        models_lightning.MODEL_NAME_FROM_CONFIG_NAME[type(model_config).__name__]
    ]

    model = model_cls.load_from_checkpoint(
        ckpt_path,
        config=model_config,
        data_spec=data_spec,
        data_config=eval_config.data_config,
        strict=False,
    )
    model.val_metrics_prefix = metric_prefix

    pred_writer = datamodule.get_pred_writer(
        pred_writer_name=eval_config.pred_writer,
        output_dir=artifact_path,
        write_interval="batch",
        output_resolution=eval_config.output_resolution,
        proj_res=eval_config.proj_res,
        prefix=metric_prefix,
        top_k=eval_config.top_k,
        ranking_metric=eval_config.ranking_metric,
        sort_dir=eval_config.sort_dir,
    )

    cbs = [pred_writer]

    if eval_config.log_masked_iou:
        robust_iou_logger = callbacks.ValMaskedIoULogger(
            prefix=metric_prefix,
            f_out=datamodule.get_classes(),
        )
        cbs.append(robust_iou_logger)

    mlf_logger = MLFlowLogger(
        experiment_name=eval_config.train_config.mlflow_expmt,
        tracking_uri=get_tracking_uri(),
    )
    mlf_logger._run_id = run_id

    # always evaluate on at most one GPU (DDP copies samples across devices to fill up batches):
    if pl_config.gpus is not None and pl_config.gpus > 0:
        print(
            "Evaluation should always be done on at most one gpu to get accurate results,",
            "setting gpus=1",
        )
        pl_config.gpus = 1
        pl_config.accelerator = None

    trainer = pl.Trainer(
        **asdict(pl_config), logger=mlf_logger, callbacks=cbs, weights_summary="full"
    )

    if eval_config.validate:
        trainer.validate(model, datamodule=datamodule)

    if eval_config.predict:
        trainer.predict(model, datamodule=datamodule)

    print("Evaluation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="run_configs/default_evaluate_run_config.py",
        help="path to config python file, relative to HEAL-SWIN path",
    )

    args = parser.parse_args()

    evaluate_config = utils.get_config_from_config_path(args.config_path, "get_eval_run_config")
    pl_config = utils.get_config_from_config_path(args.config_path, "get_pl_config")

    evaluate(evaluate_config, pl_config, args.config_path)
