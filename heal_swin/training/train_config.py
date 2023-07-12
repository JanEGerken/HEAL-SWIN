from dataclasses import dataclass, field
from typing import Optional, Literal, Union, List, Dict
from datetime import timedelta
from pathlib import Path

from pytorch_lightning.accelerators import Accelerator

from heal_swin.models_lightning.models_lightning import MODEL_CONFIGS_LITERAL
from heal_swin.models_lightning.segmentation.model_lightning_swin_hp import (
    WoodscapeSegmenterSwinHPConfig,
)

from heal_swin.data.data_config import (
    WoodscapeFlatConfig,
    WoodscapeHPConfig,
    WoodscapeDepthFlatConfig,
    WoodscapeHPDepthConfig,
)


@dataclass
class TrainConfig:
    name: str = "train_config"
    job_id: str = "no_job_id"
    description: Optional[str] = None
    ckpt_metric: str = "val_iou_global_ignored"
    ckpt_mode: str = "max"
    eval_after_train: bool = True
    mlflow_expmt: str = "woodscape_tests"
    log_gpu_stats: bool = True
    early_stopping: bool = False
    early_stopping_monitor: str = "val_iou_global_ignored"
    early_stopping_mode: str = "max"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    seed: Optional[int] = None
    load_checkpoint: Optional[str] = None
    logging_step_offset: int = 0


@dataclass
class SingleModelTrainRun:
    train: TrainConfig = field(default_factory=TrainConfig)
    data: Literal[
        WoodscapeFlatConfig,
        WoodscapeHPConfig,
        WoodscapeDepthFlatConfig,
        WoodscapeHPDepthConfig,
    ] = field(default_factory=WoodscapeHPConfig)
    model: MODEL_CONFIGS_LITERAL = field(default_factory=WoodscapeSegmenterSwinHPConfig)


@dataclass
class ResumeConfig:
    path: str  # MLFlow runid
    epoch: Optional[Literal["best", "last", "number"]] = "last"
    epoch_number: Optional[str] = None  # Optional epoch number to resume from if epoch == "number"
    train_run_config: SingleModelTrainRun = field(default_factory=SingleModelTrainRun)


@dataclass
class PLConfig:
    checkpoint_callback: bool = True
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0.0
    gradient_clip_algorithm: str = "norm"
    process_position: int = 0
    num_nodes: int = 1
    num_processes: int = 1
    gpus: Optional[Union[List[int], str, int]] = None
    auto_select_gpus: bool = False
    tpu_cores: Optional[Union[List[int], str, int]] = None
    log_gpu_memory: Optional[str] = None
    progress_bar_refresh_rate: Optional[int] = None
    overfit_batches: Union[int, float] = 0.0
    track_grad_norm: Union[int, float, str] = -1
    check_val_every_n_epoch: int = 1
    fast_dev_run: Union[int, bool] = False
    accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None
    limit_train_batches: Union[int, float] = 1.0
    limit_val_batches: Union[int, float] = 1.0
    limit_test_batches: Union[int, float] = 1.0
    limit_predict_batches: Union[int, float] = 1.0
    val_check_interval: Union[int, float] = 1.0
    flush_logs_every_n_steps: int = 100
    log_every_n_steps: int = 50
    accelerator: Optional[Union[str, Accelerator]] = None
    sync_batchnorm: bool = False
    precision: int = 32
    weights_save_path: Optional[str] = None
    num_sanity_val_steps: int = 2
    truncated_bptt_steps: Optional[int] = None
    resume_from_checkpoint: Optional[Union[Path, str]] = None
    benchmark: bool = False
    deterministic: bool = False
    reload_dataloaders_every_epoch: bool = False
    auto_lr_find: Union[bool, str] = False
    replace_sampler_ddp: bool = True
    terminate_on_nan: bool = False
    auto_scale_batch_size: Union[str, bool] = False
    prepare_data_per_node: bool = True
    amp_backend: str = "native"
    amp_level: str = "O2"
    distributed_backend: Optional[str] = None
    move_metrics_to_cpu: bool = False
    multiple_trainloader_mode: str = "max_size_cycle"
    stochastic_weight_avg: bool = False
