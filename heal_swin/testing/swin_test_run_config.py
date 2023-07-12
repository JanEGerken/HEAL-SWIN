import os

import torch

from heal_swin.training.train_config import TrainConfig, SingleModelTrainRun, PLConfig
from heal_swin.data.data_config import WoodscapeFlatConfig, WoodscapeCommonConfig
from heal_swin.models_lightning.segmentation.model_lightning_swin import (
    WoodscapeSegmenterSwinConfig,
)
from heal_swin.models_torch.swin_transformer import SwinTransformerConfig


def get_train_run_config():
    if "SLURM_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
    else:
        job_id = "flat_test_run"

    return SingleModelTrainRun(
        train=TrainConfig(
            job_id=job_id,
            mlflow_expmt="woodscape_tests",
        ),
        data=WoodscapeFlatConfig(
            common=WoodscapeCommonConfig(
                train_worker=2,
                val_worker=2,
            ),
            input_width=12,
            input_height=8,
        ),
        model=WoodscapeSegmenterSwinConfig(
            swin_transformer_config=SwinTransformerConfig(
                window_size=(4, 6),
                patch_size=(1, 1),
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
