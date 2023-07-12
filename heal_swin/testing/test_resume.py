import os
import tempfile
from pathlib import Path
import subprocess

from heal_swin.testing.test_swin import ValidateSwinMlflowTrainRun


def test_swin_resume():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpfile = Path(tmpdirname) / "slurm.out"

        train_command = (
            "python3 -m heal_swin.train --config_path heal_swin/testing/swin_test_run_config.py"
        )
        command = f"{train_command} > {tmpfile}"

        assert subprocess.run(command, shell=True).returncode == 0

        mlflow_validator = ValidateSwinMlflowTrainRun(tmpfile)
        mlflow_validator.validate_mlflow_run()

    resume_env = os.environ.copy()
    resume_env["RESUME_RUN_ID"] = mlflow_validator.run_id

    resume_command = (
        "python3 -m heal_swin.resume --config_path heal_swin/testing/resume_test_run_config.py"
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpfile = Path(tmpdirname) / "slurm.out"
        command = f"{resume_command} > {tmpfile}"

        assert subprocess.run(command, shell=True, env=resume_env).returncode == 0

        mlflow_validator = ValidateSwinMlflowTrainRun(tmpfile)
        mlflow_validator.files.remove("swin_test_run_config.py")
        mlflow_validator.files.append("resume_test_run_config.py")
        mlflow_validator.validate_mlflow_run()

    return True
