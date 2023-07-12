#!/usr/bin/env -S python3 -u
# fmt: off
#SBATCH -t 1-00:00:00  # noqa: E265
#SBATCH -o ../../../slurm/slurm-%A_%a.out  # noqa: E265
# this is needed to prevent black from formatting the above SBATCH comments...
dummy="dummy"  # noqa: E225
# fmt: on

import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

#######################################################################
# Run this file as an array job.
# For flat:
# sbatch -a 0-4 evaluate_all_config.py
# For HP:
# sbatch -a 0-5 evaluate_all_config.py
#######################################################################

RUN_ID = os.getenv("RUN_ID", default="")

EPOCH = "best"
EPOCH_NUMBER = None


def get_eval_run_config():
    from heal_swin.utils import utils
    from heal_swin.evaluation.evaluate_config import EvaluateConfig
    from heal_swin.data.data_config import WoodscapeFlatConfig, WoodscapeHPConfig

    train_run_config = utils.load_config(RUN_ID, "run_config")
    data_config = train_run_config.data
    train_config = train_run_config.train

    task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")
    if isinstance(data_config, WoodscapeFlatConfig):
        flat_hp = "flat"
        if task_count != "5":
            print(f"\n\nWARNING: found {task_count} tasks, expected 5\n\n")
    elif isinstance(data_config, WoodscapeHPConfig):
        flat_hp = "hp"
        if task_count != "6":
            print(f"\n\nWARNING: found {task_count} tasks, expected 6\n\n")

    if EPOCH in ["best", "last"]:
        metric_prefix = EPOCH
    elif EPOCH == "number":
        metric_prefix = f"epoch_{EPOCH_NUMBER}"

    print(40 * "-")
    print(f"Evaluating RUN_ID: {RUN_ID} on {metric_prefix} epoch.")
    print(40 * "-")

    # default values for some rarely changed parameters
    ranking_metric = "iou_ignored"
    proj_res = 966
    pred_part = "val"
    pred_samples = 1.0
    predict = True
    validate = False

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    job_id = f"{os.environ.get('SLURM_ARRAY_JOB_ID', 'no_job_id')}_{task_id}"
    if task_id == "0":
        eval_config_name = f"{metric_prefix}_validation_{job_id}"
        pred_writer = "base_writer"
        pred_samples = 10
        validate = True
    elif task_id == "1":
        eval_config_name = f"{metric_prefix}_val_best_worst_{job_id}"
        pred_writer = "best_worst_preds"
    elif task_id == "2":
        eval_config_name = f"{metric_prefix}_train_best_worst_{job_id}"
        pred_writer = "best_worst_preds"
        pred_part = "train"

    if flat_hp == "flat":
        if task_id == "3":
            eval_config_name = f"{metric_prefix}_projected_to_hp_{job_id}"
            pred_writer = "val_on_hp_projected"
        elif task_id == "4":
            eval_config_name = f"{metric_prefix}_hp_masked_iou_{job_id}"
            pred_writer = "hp_masked_iou"

    if flat_hp == "hp":
        if task_id == "3":
            eval_config_name = f"{metric_prefix}_back_projected_full_res_{job_id}"
            pred_writer = "val_on_back_projected"
        if task_id == "4":
            eval_config_name = f"{metric_prefix}_back_projected_flat_res_{job_id}"
            pred_writer = "val_on_back_projected"
            proj_res = (640, 768)
        if task_id == "5":
            eval_config_name = f"{metric_prefix}_back_projected_hp_masked_iou_flat_res_{job_id}"
            pred_writer = "back_projected_hp_masked_iou"
            proj_res = (640, 768)

    data_config.common.pred_samples = pred_samples
    data_config.predict_part = pred_part
    return EvaluateConfig(
        path=RUN_ID,
        epoch=EPOCH,
        epoch_number=EPOCH_NUMBER,
        eval_config_name=eval_config_name,
        metric_prefix=metric_prefix,
        override_eval_config=True,
        ranking_metric=ranking_metric,
        pred_writer=pred_writer,
        predict=predict,
        validate=validate,
        log_masked_iou=True,
        top_k=5,
        sort_dir="asc",
        proj_res=proj_res,
        data_config=data_config,
        train_config=train_config,
    )


def get_pl_config():
    from heal_swin.utils import utils
    from heal_swin.training.train_config import PLConfig

    import torch

    try:
        train_pl_config = utils.load_config(RUN_ID, "pl_config")
    except AssertionError:  # thrown by file not found in load_config
        train_args_dict = utils.load_config(RUN_ID, "args")
        train_pl_config = PLConfig()
        for key, value in train_args_dict.items():
            if hasattr(train_pl_config, key):
                setattr(train_pl_config, key, value)

    if torch.cuda.is_available():
        train_pl_config.gpus = 1
    else:
        train_pl_config.gpus = 0
    return train_pl_config


def main():
    this_path = str(Path(__file__).absolute())

    if "SLURM_SUBMIT_DIR" in os.environ:
        base_path = str(Path(os.environ["SLURM_SUBMIT_DIR"]).parents[2])
    else:
        base_path = str(Path(this_path).parents[3])

    run_py_path = os.path.join(base_path, "run.py")

    command = ["python3", "-u", run_py_path]
    command += ["--env", "singularity"]
    command += ["evaluate"]
    command += ["--config_path", this_path]
    print(" ".join(command))

    subprocess.run(command)


if __name__ == "__main__":
    main()
