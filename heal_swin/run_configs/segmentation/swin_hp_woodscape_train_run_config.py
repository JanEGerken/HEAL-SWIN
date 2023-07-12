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


def get_train_run_config():
    from heal_swin.training.train_config import SingleModelTrainRun, TrainConfig
    from heal_swin.data.data_config import WoodscapeHPConfig, WoodscapeCommonConfig
    from heal_swin.models_lightning.segmentation.model_lightning_swin_hp import (
        WoodscapeSegmenterSwinHPConfig,
    )
    from heal_swin.models_torch.swin_hp_transformer import SwinHPTransformerConfig, UnetDecoder
    from heal_swin.training.optimizer import OptimizerConfig

    if "SLURM_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
    else:
        job_id = "no_job_id"

    train_config = TrainConfig(
        job_id=job_id,
        mlflow_expmt="woodscape",
        description="swin-hp",
        eval_after_train=False,
        early_stopping=False,
    )
    data_config = WoodscapeHPConfig(
        common=WoodscapeCommonConfig(
            version="woodscape",
            batch_size=2,
            val_batch_size=4,
            pred_batch_size=4,
            train_worker=5,
            val_worker=5,
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
            0.34876218,
            0.44370147,
            0.89928661,
            1.1544441,
            1.3193849,
            1.7061983,
            0.73121492,
            1.2444171,
            1.6885881,
            2.364354,
        ],
    )

    return SingleModelTrainRun(train=train_config, data=data_config, model=model_config)


def get_pl_config():
    from heal_swin.training.train_config import PLConfig

    return PLConfig(
        max_epochs=1000,
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
