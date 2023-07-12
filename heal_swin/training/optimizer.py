#!/usr/bin/env python
from dataclasses import dataclass
from typing import Optional, Literal

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR


class LightningReduceLROnPlateau:
    def __init__(self, config, optimizer):
        """config: OptimizerConfig"""
        self.config = config
        self.optimizer = optimizer

    def get_scheduler_dict(self):
        sched_dict = {
            "scheduler": ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.scheduler_mode,
                patience=self.config.scheduler_patience,
                threshold=self.config.scheduler_threshold,
                factor=self.config.scheduler_factor,
                min_lr=self.config.scheduler_min_lr,
            ),
            "monitor": self.config.scheduler_monitor,
        }
        return sched_dict


class LightningExponentialLR:
    def __init__(self, config, optimizer):
        """config: OptimizerConfig"""
        self.config = config
        self.optimizer = optimizer

    def get_scheduler_dict(self):
        sched_dict = {
            "scheduler": ExponentialLR(self.optimizer, gamma=self.config.scheduler_factor)
        }
        return sched_dict


@dataclass
class OptimizerConfig:
    optimizer_name: Literal["Adam", "AdamW"] = "Adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    scheduler: Optional[Literal[LightningReduceLROnPlateau, LightningReduceLROnPlateau]] = None
    scheduler_mode: str = "min"
    scheduler_patience: int = 10
    scheduler_threshold: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-5
    scheduler_monitor: str = "train_loss"


def get_lightning_optimizer_dict(params, config):
    OPTIMIZER_CLASSES = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}
    optimizer = OPTIMIZER_CLASSES[config.optimizer_name](
        params, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    opt_dict = {"optimizer": optimizer}
    if config.scheduler is not None:
        scheduler = config.scheduler(config, optimizer)
        opt_dict["lr_scheduler"] = scheduler.get_scheduler_dict()
    return opt_dict
