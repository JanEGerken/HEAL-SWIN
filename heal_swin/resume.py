#!/usr/bin/env python3
import argparse
import os
import shutil

from heal_swin.models_lightning import models_lightning
from heal_swin.utils import get_paths, utils
from heal_swin.train import train_model


def main(config_path, pl_config, resume_config):
    ckpt_path, artifact_path, _ = utils.check_and_get_ckpt_paths(
        resume_config.path, resume_config.epoch, resume_config.epoch_number
    )

    pl_config.resume_from_checkpoint = ckpt_path

    abs_config_path = get_paths.get_abs_path_from_config_path(config_path)
    config_name = os.path.basename(abs_config_path)
    shutil.copyfile(abs_config_path, os.path.join(artifact_path, config_name))

    train_model(
        config_path=config_path,
        train_config=resume_config.train_run_config.train,
        pl_config=pl_config,
        model_class=models_lightning.MODEL_FROM_CONFIG_NAME[
            resume_config.train_run_config.model.__class__.__name__
        ],
        model_config=resume_config.train_run_config.model,
        data_config=resume_config.train_run_config.data,
        run_config=resume_config.train_run_config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="run_configs/default_resume_config.py",
        help="path to config python file, relative to HEAL-SWIN path",
    )

    args = parser.parse_args()

    resume_config = utils.get_config_from_config_path(args.config_path, "get_resume_run_config")
    pl_config = utils.get_config_from_config_path(args.config_path, "get_pl_config")
    main(args.config_path, pl_config, resume_config)
