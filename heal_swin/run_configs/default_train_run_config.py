#!/usr/bin/env -S python3 -u
# fmt: off
#SBATCH -t 4-00:00:00  # noqa: E265
#SBATCH -o ../../slurm/slurm-%j.out  # for array jobs, this should be slurm-%A_%a.out # noqa: E265
# this is needed to prevent black from formatting the above SBATCH comments...
dummy="dummy"  # noqa: E225
# fmt: on

import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402


def get_train_run_config():
    from heal_swin.training.train_config import SingleModelTrainRun, TrainConfig

    if "SLURM_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
    else:
        job_id = "no_job_id"

    return SingleModelTrainRun(train=TrainConfig(job_id=job_id))


def get_pl_config():
    from heal_swin.training.train_config import PLConfig

    return PLConfig()


def main():
    this_path = str(Path(__file__).absolute())

    if "SLURM_SUBMIT_DIR" in os.environ:
        base_path = str(Path(os.environ["SLURM_SUBMIT_DIR"]).parents[1])
    else:
        base_path = str(Path(this_path).parents[2])

    run_py_path = os.path.join(base_path, "run.py")
    command = ["python3", "-u", run_py_path]
    command += ["--env", "singularity"]
    command += ["train"]
    command += ["--config_path", this_path]
    print(" ".join(command))

    subprocess.run(command)


if __name__ == "__main__":
    main()
