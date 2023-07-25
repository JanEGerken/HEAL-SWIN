#!/usr/bin/env -S python3 -u
# fmt: off
#SBATCH -t 7-00:00:00  # noqa: E265
#SBATCH -o ../../../slurm/slurm-%j.out  # for array jobs, this should be slurm-%A_%a.out # noqa: E265
# this is needed to prevent black from formatting the above SBATCH comments...
dummy="dummy"  # noqa: E225
# fmt: on

import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import argparse
import math

def get_argparser_for_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_data_fraction",
        type=float,
        default=0.1,
        help="Fraction of training data used.",
    )
    parser.add_argument(
        "--data_fraction_seed",
        type=int,
        default=2,
        help="Seed for determining subset of training data.",
    )

    return parser

def get_train_run_config(training_data_fraction=0.1, data_fraction_seed=2, **kwargs):
    from heal_swin.training.train_config import SingleModelTrainRun, TrainConfig
    from heal_swin.data.data_config import WoodscapeFlatConfig, WoodscapeCommonConfig
    from heal_swin.models_lightning.segmentation.model_lightning_swin import (
        WoodscapeSegmenterSwinConfig,
    )
    from heal_swin.models_torch.swin_transformer import SwinTransformerConfig
    from heal_swin.training.optimizer import OptimizerConfig

    if "SLURM_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
    else:
        job_id = "no_job_id"

    train_config = TrainConfig(
        job_id=job_id,
        mlflow_expmt="data_eff_synwoodscape_large_plus_AD",
        description=f"Data Efficiency SWIN-Unet (Flat): Fraction: {training_data_fraction}, Seed: {data_fraction_seed}",
        eval_after_train=False,
        early_stopping=False,
    )
    data_config = WoodscapeFlatConfig(
        common=WoodscapeCommonConfig(
            version="synwoodscape_large_plus_AD",
            batch_size=2,
            val_batch_size=4,
            pred_batch_size=4,
            train_worker=5,
            val_worker=5,
            training_data_fraction=training_data_fraction,
            data_fraction_seed=data_fraction_seed
        ),
        input_width=768,
        input_height=640,
        padding=(0, 0, 0, 0),
    )
    model_config = WoodscapeSegmenterSwinConfig(
        swin_transformer_config=SwinTransformerConfig(
            window_size=8,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            patch_size=2,
            shift_size=2,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            ape=False,
            use_cos_attn=True,
            use_v2_norm_placement=True,
        ),
        optimizer_config=OptimizerConfig(
            scheduler=None,
            learning_rate=0.000954993,
        ),
        class_weights=[
            0.64988532,
            0.5111932,
            1.18160048,
            0.88803174,
            0.39921158,
            0.75032628,
            0.88420746,
            1.91088558,
            0.67315916,
            2.34623503,
            1.67844596,
            0.43332322,
        ],
    )

    return SingleModelTrainRun(train=train_config, data=data_config, model=model_config)


def get_pl_config(training_data_fraction=0.1, **kwargs):
    from heal_swin.training.train_config import PLConfig

    return PLConfig(
        max_epochs=math.ceil(500 / training_data_fraction),
        gpus=4,
        accelerator="ddp",
    )


def main(raw_args=None):
    this_path = str(Path(__file__).absolute())

    if "SLURM_SUBMIT_DIR" in os.environ:
        base_path = str(Path(os.environ["SLURM_SUBMIT_DIR"]).parents[2])
    else:
        base_path = str(Path(this_path).parents[3])

    run_py_path = os.path.join(base_path, "run.py")
    command = ["python3", "-u", run_py_path]
    command += ["--env", "singularity"]
    command += ["train"]
    command += ["--config_path", this_path]

    parser = argparse.ArgumentParser()
    _, unknown = parser.parse_known_args(raw_args)

    command += unknown

    print(" ".join(command))

    subprocess.run(command)


if __name__ == "__main__":
    main()
