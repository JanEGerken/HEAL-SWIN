import os

from heal_swin.utils import utils
from heal_swin.training.train_config import ResumeConfig

RUN_ID = os.environ["RESUME_RUN_ID"]


def get_resume_run_config():

    train_run_config = utils.load_config(RUN_ID, "run_config")

    return ResumeConfig(path=RUN_ID, epoch="last", train_run_config=train_run_config)


def get_pl_config():
    previous_pl_config = utils.load_config(RUN_ID, "pl_config")
    previous_pl_config.max_epochs = 2
    return previous_pl_config
