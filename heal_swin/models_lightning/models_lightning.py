from typing import Literal

from heal_swin.models_lightning.segmentation import (
    model_lightning_swin,
    model_lightning_swin_hp,
)
from heal_swin.models_lightning.depth_estimation import (
    model_lightning_depth_swin,
    model_lightning_depth_swin_hp,
)

MODEL_CLASSES = [
    model_lightning_swin.WoodscapeSegmenterSwin,
    model_lightning_swin_hp.WoodscapeSegmenterSwinHP,
    model_lightning_depth_swin.WoodscapeDepthSwin,
    model_lightning_depth_swin_hp.WoodscapeDepthSwinHP,
]

MODEL_CONFIGS_LITERAL = Literal[
    model_lightning_swin.WoodscapeSegmenterSwin.CONFIG_CLASS,
    model_lightning_swin_hp.WoodscapeSegmenterSwinHP.CONFIG_CLASS,
    model_lightning_depth_swin.WoodscapeDepthSwin.CONFIG_CLASS,
    model_lightning_depth_swin_hp.WoodscapeDepthSwinHP.CONFIG_CLASS,
]

MODELS = {model.NAME: model for model in MODEL_CLASSES}

MODEL_NAME_FROM_CONFIG_NAME = {
    model.CONFIG_CLASS.__name__: model.__name__ for model in MODEL_CLASSES
}

MODEL_FROM_CONFIG_NAME = {model.CONFIG_CLASS.__name__: model for model in MODEL_CLASSES}

MODEL_FROM_NAME = {model.__name__: model for model in MODEL_CLASSES}
