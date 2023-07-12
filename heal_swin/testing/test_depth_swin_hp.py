import tempfile
import subprocess
from pathlib import Path

from heal_swin.testing.validate_mlflow import ValidateMlflowTrainRun


class ValidateDepthSwinHPMlflowTrainRun(ValidateMlflowTrainRun):
    def __init__(self, slurm_path):
        super().__init__(slurm_path)
        self.params += [
            "dim_in",
            "optimizer_config.scheduler_factor",
            "optimizer_config.scheduler_min_lr",
            "optimizer_config.scheduler_mode",
            "optimizer_config.scheduler_monitor",
            "optimizer_config.scheduler_patience",
            "optimizer_config.scheduler_threshold",
            "optimizer_config.scheduler",
            "swin_hp_transformer_config.ape",
            "swin_hp_transformer_config.attn_drop_rate",
            "swin_hp_transformer_config.depths",
            "swin_hp_transformer_config.drop_path_rate",
            "swin_hp_transformer_config.drop_rate",
            "swin_hp_transformer_config.embed_dim",
            "swin_hp_transformer_config.mlp_ratio",
            "swin_hp_transformer_config.norm_layer",
            "swin_hp_transformer_config.num_heads",
            "swin_hp_transformer_config.patch_embed_norm_layer",
            "swin_hp_transformer_config.patch_norm",
            "swin_hp_transformer_config.patch_size",
            "swin_hp_transformer_config.qk_scale",
            "swin_hp_transformer_config.qkv_bias",
            "swin_hp_transformer_config.shift_size",
            "swin_hp_transformer_config.use_checkpoint",
            "swin_hp_transformer_config.window_size",
            "use_logvar",
            "loss",
            "data_transform",
            "normalize_data",
            "mask_background",
        ]

        self.metrics = [
            "epoch",
            "train_loss_epoch",
            "val_loss",
            "train_mse",
            "val_mse",
        ]
        self.files += ["depth_swin_hp_test_run_config.py"]


def test_swin_training():

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpfile = Path(tmpdirname) / "slurm.out"

        command = (
            "python3 -m heal_swin.train --config_path "
            "heal_swin/testing/depth_swin_hp_test_run_config.py"
        )
        command += f" > {tmpfile}"

        process = subprocess.run(command, shell=True)
        assert process.returncode == 0

        mlflow_validator = ValidateDepthSwinHPMlflowTrainRun(tmpfile)
        mlflow_validator.validate_mlflow_run()

    return True
