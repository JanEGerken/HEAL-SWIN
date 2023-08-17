#!/usr/bin/env -S python3 -u
# fmt: off
#SBATCH -t 7-00:00:00  # noqa: E265
#SBATCH -o ../../../slurm/slurm-%A_%a.out  # noqa: E265
# this is needed to prevent black from formatting the above SBATCH comments...
dummy="dummy"  # noqa: E225
# fmt: on

import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

#######################################################################
# Run this file as an array job.
# sbatch -a 0-8 swin_hp_synwoodscape_large_plus_AD_data_efficiency_train_run_config.py

# 8 is len(TRAINING_DATA_FRACTIONS) - 1
#######################################################################

import math

TRAINING_DATA_FRACTIONS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
DATA_FRACTION_SEED = 2  # 3


def get_train_run_config():
    from heal_swin.training.train_config import SingleModelTrainRun, TrainConfig
    from heal_swin.data.data_config import WoodscapeHPConfig, WoodscapeCommonConfig
    from heal_swin.models_lightning.segmentation.model_lightning_swin_hp import (
        WoodscapeSegmenterSwinHPConfig,
    )
    from heal_swin.models_torch.swin_hp_transformer import (
        SwinHPTransformerConfig,
        UnetDecoder,
    )
    from heal_swin.training.optimizer import OptimizerConfig

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    job_id = f"{os.environ.get('SLURM_ARRAY_JOB_ID', 'no_job_id')}_{task_id}"

    assert task_id < len(
        TRAINING_DATA_FRACTIONS
    ), f"Invalid ARRAY_TASK_ID: {task_id} >= {len(TRAINING_DATA_FRACTIONS)} (len(TRAINING_DATA_FRACTIONS))"

    training_data_fraction = TRAINING_DATA_FRACTIONS[task_id]
    data_fraction_seed = DATA_FRACTION_SEED

    train_config = TrainConfig(
        job_id=job_id,
        mlflow_expmt="data_eff_synwoodscape_large_plus_AD",
        description=f"Data Efficiency SWIN-HP: Fraction: {training_data_fraction}, Seed: {data_fraction_seed}",
        eval_after_train=False,
        early_stopping=False,
    )
    data_config = WoodscapeHPConfig(
        common=WoodscapeCommonConfig(
            version="synwoodscape_large_plus_AD",
            batch_size=2,
            val_batch_size=4,
            pred_batch_size=4,
            train_worker=5,
            val_worker=5,
            training_data_fraction=training_data_fraction,
            data_fraction_seed=data_fraction_seed,
        ),
        input_nside=256,
        input_base_pix=8,
    )
    model_config = WoodscapeSegmenterSwinHPConfig(
        swin_hp_transformer_config=SwinHPTransformerConfig(
            window_size=64,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            patch_size=4,
            shift_size=4,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            ape=False,
            use_cos_attn=True,
            use_v2_norm_placement=True,
            shift_strategy="ring_shift",
            rel_pos_bias="flat",
            decoder_class=UnetDecoder,
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


def get_pl_config():
    from heal_swin.training.train_config import PLConfig

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    training_data_fraction = TRAINING_DATA_FRACTIONS[task_id]

    return PLConfig(
        max_epochs=math.ceil(500 / training_data_fraction),
        gpus=4,
        accelerator="ddp",
    )


def main():
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

    print(" ".join(command))

    subprocess.run(command)


if __name__ == "__main__":
    main()
