from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal, Union

from heal_swin.training.train_config import TrainConfig

from heal_swin.data.data_config import (
    WoodscapeFlatConfig,
    WoodscapeHPConfig,
    WoodscapeHPDepthConfig,
    WoodscapeDepthFlatConfig,
)


@dataclass
class EvaluateConfig:
    path: str
    # A run identifier, either an mlflow run_id or full path to a ckpt
    eval_config_name: str = "best"
    # The evaluate config will be saved under this name
    epoch: Optional[Literal["best", "last", "number"]] = "best"
    # If not an explicit ckpt path was specified, this allows for loading the matching saved ckpt
    epoch_number: Optional[str] = None
    # If "number" was specified for "epoch" enter an epoch number to evaluate here
    metric_prefix: Optional[str] = None
    # This will be prepended to all saved predictions and metrics
    override_eval_config: bool = False
    # If this is given and "eval_config_name" agrees with an already existing eval_config the
    # program will override the old eval_config
    pred_writer: Optional[str] = None
    validate: bool = True
    predict: bool = True
    # For back-projected HP: resolution of back-projected image. If one entry, rescale shorter side
    # to this, if two entries, target resolution
    proj_res: Union[int, Tuple[int, int]] = 966
    output_resolution: float = 0.5  # resolution of plotted images, no effect on metrics
    top_k: int = 5  # how many best/worst predictions to save
    ranking_metric: Literal["acc", "iou", "acc_ignored", "iou_ignored", "mse"] = "iou_ignored"
    sort_dir: Literal["asc", "desc"] = "asc"  # asc: best have highest metric value
    log_masked_iou: bool = False
    train_config: TrainConfig = field(default_factory=TrainConfig)
    data_config: Literal[
        WoodscapeFlatConfig,
        WoodscapeHPConfig,
        WoodscapeDepthFlatConfig,
        WoodscapeHPDepthConfig,
    ] = field(default_factory=WoodscapeFlatConfig)
