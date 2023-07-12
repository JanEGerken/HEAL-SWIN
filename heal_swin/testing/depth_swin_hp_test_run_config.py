import os

import torch

from heal_swin.training.train_config import TrainConfig, SingleModelTrainRun, PLConfig
from heal_swin.data.data_config import (
    WoodscapeHPDepthConfig,
    WoodscapeCommonConfig,
    WoodscapeDepthCommonConfig,
)
from heal_swin.models_lightning.depth_estimation.model_lightning_depth_swin_hp import (
    WoodscapeDepthSwinHPConfig,
)
from heal_swin.models_lightning.depth_estimation.depth_common_config import CommonDepthConfig
from heal_swin.models_torch.swin_hp_transformer import SwinHPTransformerConfig


def get_train_run_config():
    if "SLURM_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
    else:
        job_id = "flat_test_run"

    return SingleModelTrainRun(
        train=TrainConfig(
            job_id=job_id,
            mlflow_expmt="woodscape_tests",
            early_stopping_monitor="val_mse",
            early_stopping_mode="min",
            ckpt_metric="val_mse",
            ckpt_mode="min",
        ),
        data=WoodscapeHPDepthConfig(
            common=WoodscapeCommonConfig(
                train_worker=2,
                val_worker=2,
            ),
            common_depth=WoodscapeDepthCommonConfig(
                mask_background=True,
                normalize_data="standardize",
                data_transform=None,
            ),
            input_nside=32,
            input_base_pix=8,
        ),
        model=WoodscapeDepthSwinHPConfig(
            swin_hp_transformer_config=SwinHPTransformerConfig(
                window_size=4,
                patch_size=4,
                depths=(2, 1),
                num_heads=(1, 1),
                embed_dim=2,
            ),
            common_depth_config=CommonDepthConfig(
                loss="l2",
                use_logvar=False,
            ),
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
