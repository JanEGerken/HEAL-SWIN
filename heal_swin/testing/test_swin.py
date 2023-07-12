import tempfile
from pathlib import Path
import subprocess

from heal_swin.testing.validate_mlflow import ValidateMlflowTrainRun


class ValidateSwinMlflowTrainRun(ValidateMlflowTrainRun):
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
            "swin_transformer_config.ape",
            "swin_transformer_config.attn_drop_rate",
            "swin_transformer_config.depths",
            "swin_transformer_config.drop_path_rate",
            "swin_transformer_config.drop_rate",
            "swin_transformer_config.embed_dim",
            "swin_transformer_config.final_upsample",
            "swin_transformer_config.mlp_ratio",
            "swin_transformer_config.num_heads",
            "swin_transformer_config.patch_embed_norm_layer",
            "swin_transformer_config.patch_norm",
            "swin_transformer_config.patch_size",
            "swin_transformer_config.qk_scale",
            "swin_transformer_config.qkv_bias",
            "swin_transformer_config.shift_size",
            "swin_transformer_config.use_checkpoint",
            "swin_transformer_config.window_size",
        ]
        self.files += ["swin_test_run_config.py"]


def test_swin_training():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpfile = Path(tmpdirname) / "slurm.out"

        command = (
            "python3 -m heal_swin.train --config_path heal_swin/testing/swin_test_run_config.py"
        )

        process = subprocess.run(command, capture_output=True, shell=True)
        assert process.returncode == 0, process.stdout

        with open(tmpfile, "w") as f:
            f.write(process.stdout.decode("utf-8"))

        mlflow_validator = ValidateSwinMlflowTrainRun(tmpfile)
        mlflow_validator.validate_mlflow_run()

    return True
