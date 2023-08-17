from dataclasses import dataclass, field
from typing import Optional, Union, List, Literal


@dataclass
class DataCommonConfig:
    train_worker: int = 2
    val_worker: int = 2
    shuffle: bool = True
    batch_size: int = 32
    val_batch_size: int = 32
    pred_batch_size: int = 4
    manual_overfit_batches: int = 0

    training_data_fraction: float = 1.0  # fraction of data to use for training
    data_fraction_seed: int = (
        42  # enables fixing the subset taken as fraction for comparing multiple runs
    )

    def __post_init__(self):
        assert (
            0.0 < self.training_data_fraction <= 1.0
        ), "training_data_fraction not in (0.0, 1.0]"


@dataclass
class WoodscapeCommonConfig(DataCommonConfig):
    pred_samples: Union[int, float] = 10  # if float: fraction of val/train data
    rotate_pole: bool = False
    s2_bkgd_class: int = 0
    seed: Optional[int] = 42
    cam_pos: Optional[Literal["fv", "rv", "mvl", "mvr"]] = None
    train_share: float = 0.8
    crop_green: bool = False
    version: str = "woodscape"


@dataclass
class WoodscapeFlatConfig:
    common: WoodscapeCommonConfig = field(default_factory=WoodscapeCommonConfig)
    pred_part: Literal["train", "val"] = "val"  # on which part of the data to predict
    input_width: int = 768
    input_height: int = 640
    padding: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    shuffle_train_val_split: bool = True  # whether to shuffle the data before splitting
    nside: int = 256  # nside parameter of HP dataset used for projected predictions
    base_pix: int = 8  # base_pix parameter of HP dataset used for projected predictions


@dataclass
class WoodscapeHPConfig:
    common: WoodscapeCommonConfig = field(default_factory=WoodscapeCommonConfig)
    pred_part: Literal["train", "val"] = "val"  # on which part of the data to predict
    input_nside: int = 256
    input_base_pix: int = 8
    shuffle_train_val_split: bool = True  # whether to shuffle the data before splitting


@dataclass
class WoodscapeDepthCommonConfig:
    mask_background: bool = False
    data_transform: Optional[Literal["log", "inv", "None"]] = "None"
    normalize_data: Optional[Literal["standardize", "min-max", "None"]] = "None"


@dataclass
class WoodscapeDepthFlatConfig:
    common: WoodscapeCommonConfig = field(default_factory=WoodscapeCommonConfig)
    common_depth: WoodscapeDepthCommonConfig = field(
        default_factory=WoodscapeDepthCommonConfig
    )
    pred_part: Literal["train", "val"] = "val"  # on which part of the data to predict
    input_bandwidth: int = 64
    padding: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    shuffle_train_val_split: bool = True  # whether to shuffle the data before splitting
    nside: int = 256  # nside parameter of HP dataset used for projected predictions
    base_pix: int = 8  # base_pix parameter of HP dataset used for projected predictions


@dataclass
class WoodscapeHPDepthConfig:
    common: WoodscapeCommonConfig = WoodscapeCommonConfig
    common_depth: WoodscapeDepthCommonConfig = WoodscapeDepthCommonConfig
    pred_part: Literal["train", "val"] = "val"  # on which part of the data to predict
    input_nside: int = 256
    input_base_pix: int = 8
    shuffle_train_val_split: bool = True  # whether to shuffle the data before splitting
