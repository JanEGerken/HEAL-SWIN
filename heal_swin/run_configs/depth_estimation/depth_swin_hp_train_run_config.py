#!/usr/bin/env -S python3 -u
# fmt: off
#SBATCH -t 7-00:00:00  # noqa: E265
#SBATCH -o ../../../slurm/slurm-%j.out  # noqa: E265
# For single runs use slurm-%j.out  # noqa: E265
# For array jobs, this should be slurm-%A_%a.out  # noqa: E265
dummy="dummy"  # noqa: E225
# fmt: on

import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402


def get_train_run_config():
    from heal_swin.training.train_config import SingleModelTrainRun, TrainConfig
    from heal_swin.data.data_config import (
        WoodscapeCommonConfig,
        WoodscapeHPDepthConfig,
        WoodscapeDepthCommonConfig,
    )
    from heal_swin.models_lightning.depth_estimation.depth_common_config import CommonDepthConfig
    from heal_swin.models_torch.swin_hp_transformer import SwinHPTransformerConfig, UnetDecoder
    from heal_swin.models_lightning.depth_estimation.model_lightning_depth_swin_hp import (
        WoodscapeDepthSwinHPConfig,
    )
    from heal_swin.training.optimizer import OptimizerConfig

    if "SLURM_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
    else:
        job_id = "no_job_id"

    data_transform = None
    learning_rate = 0.005
    loss = "l2"
    mask_background = True
    normalize_data = "standardize"
    description = (
        "SWIN HP full run with 0.005 learning rate, l2 loss, "
        "standardize normalization, None data transformation"
    )

    train_config = TrainConfig(
        ckpt_metric="val_mse",
        ckpt_mode="min",
        description=description,
        early_stopping=False,
        early_stopping_mode="min",
        early_stopping_monitor="val_mse",
        eval_after_train=False,
        job_id=job_id,
        mlflow_expmt="depth_estimation",
    )

    data_config = WoodscapeHPDepthConfig(
        common=WoodscapeCommonConfig(
            train_worker=5,
            val_worker=5,
            shuffle=True,
            batch_size=2,
            val_batch_size=4,
            pred_batch_size=4,
            manual_overfit_batches=0,
            pred_samples=10,
            rotate_pole=False,
            seed=42,
            cam_pos=None,
            train_share=0.8,
            crop_green=False,
            version="woodscape",
        ),
        common_depth=WoodscapeDepthCommonConfig(
            mask_background=mask_background,
            data_transform=data_transform,
            normalize_data=normalize_data,
        ),
        pred_part="val",
        input_nside=256,
        input_base_pix=8,
        shuffle_train_val_split=True,
    )
    model_config = WoodscapeDepthSwinHPConfig(
        swin_hp_transformer_config=SwinHPTransformerConfig(
            ape=False,
            attn_drop_rate=0.1,
            decoder_class=UnetDecoder,
            depths=[2, 2, 6, 2],
            drop_path_rate=0.1,
            drop_rate=0.1,
            embed_dim=96,
            mlp_ratio=4.0,
            norm_layer="LayerNorm",
            num_heads=[3, 6, 12, 24],
            patch_embed_norm_layer=None,
            patch_norm=True,
            patch_size=4,
            qk_scale=None,
            qkv_bias=True,
            rel_pos_bias="flat",
            shift_size=4,
            shift_strategy="ring_shift",
            use_cos_attn=True,
            use_v2_norm_placement=True,
            window_size=64,
        ),
        optimizer_config=OptimizerConfig(
            learning_rate=learning_rate,
            scheduler=None,
            weight_decay=0.0,
        ),
        common_depth_config=CommonDepthConfig(
            loss=loss, use_logvar=False, train_uncertainty_after=None, huber_delta=1
        ),
    )

    return SingleModelTrainRun(train=train_config, data=data_config, model=model_config)


def get_pl_config():
    from heal_swin.training.train_config import PLConfig

    return PLConfig(
        max_epochs=1000,
        gpus=4,
        accelerator="ddp",
        gradient_clip_val=0,
        gradient_clip_algorithm="norm",
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
