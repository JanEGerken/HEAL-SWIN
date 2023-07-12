import os

import torch

from heal_swin.training.train_config import TrainConfig, SingleModelTrainRun, PLConfig
from heal_swin.data.data_config import WoodscapeHPConfig, WoodscapeCommonConfig
from heal_swin.models_lightning.segmentation.model_lightning_swin_hp import (
    WoodscapeSegmenterSwinHPConfig,
)
from heal_swin.models_torch.swin_hp_transformer import SwinHPTransformerConfig


def get_train_run_config():
    if "SLURM_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
    else:
        job_id = "hp_test_run"

    return SingleModelTrainRun(
        train=TrainConfig(
            job_id=job_id,
            mlflow_expmt="woodscape_tests",
        ),
        data=WoodscapeHPConfig(
            common=WoodscapeCommonConfig(
                train_worker=2,
                val_worker=2,
            ),
            input_nside=32,
            input_base_pix=8,
        ),
        model=WoodscapeSegmenterSwinHPConfig(
            swin_hp_transformer_config=SwinHPTransformerConfig(
                window_size=4,
                patch_size=4,
                depths=(2, 1),
                num_heads=(1, 1),
                embed_dim=2,
            )
        ),
    )


def get_pl_config():
    gpus = min(torch.cuda.device_count(), 1)
    accelerator = "ddp_spawn" if gpus > 0 else None
    return PLConfig(
        limit_train_batches=1,
        limit_val_batches=1,
        limit_predict_batches=1,
        max_epochs=1,
        log_every_n_steps=1,
        gpus=gpus,
        accelerator=accelerator,
    )
