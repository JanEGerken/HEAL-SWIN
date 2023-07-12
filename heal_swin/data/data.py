from typing import Tuple

from heal_swin.data.segmentation import flat_datamodule, hp_datasets

from heal_swin.data.depth_estimation import flat_depth_datamodule, hp_depth_datasets
from heal_swin.data.depth_estimation.flat_depth_datamodule import WoodscapeFlatDepthDataModule
from heal_swin.data.depth_estimation.hp_depth_datasets import WoodscapeHPDepthDataModule

from heal_swin.data.data_config import WoodscapeFlatConfig, WoodscapeHPConfig
from heal_swin.data.segmentation.data_spec import create_dataspec_from_data_module

from heal_swin.data.data_config import WoodscapeDepthFlatConfig, WoodscapeHPDepthConfig
from heal_swin.data.depth_estimation.data_spec_depth import DepthDataSpec
from heal_swin.data.depth_estimation.data_spec_depth import create_depth_dataspec_from_data_module


def get_flat_data_module(config):
    dm = flat_datamodule.WoodscapeFlatSegmentationDataModule(
        pred_part=config.pred_part,
        padding=config.padding,
        size=(config.input_height, config.input_width),
        shuffle_train_val_split=config.shuffle_train_val_split,
        nside=config.nside,
        base_pix=config.base_pix,
        **config.common.__dict__,
    )
    data_spec = create_dataspec_from_data_module(dm)
    return dm, data_spec


def get_hp_data_module(config):
    dm = hp_datasets.WoodscapeHPSegmentationDataModule(
        pred_part=config.pred_part,
        nside=config.input_nside,
        base_pix=config.input_base_pix,
        shuffle_train_val_split=config.shuffle_train_val_split,
        **config.common.__dict__,
    )
    data_spec = create_dataspec_from_data_module(dm, base_pix=config.input_base_pix)
    return dm, data_spec


def get_depth_flat_data_module(config) -> Tuple[WoodscapeFlatDepthDataModule, DepthDataSpec]:
    dm = flat_depth_datamodule.WoodscapeFlatDepthDataModule(
        pred_part=config.pred_part,
        padding=config.padding,
        bandwidth=config.input_bandwidth,
        size=config.input_bandwidth * 2,
        shuffle_train_val_split=config.shuffle_train_val_split,
        nside=config.nside,
        base_pix=config.base_pix,
        data_transform=config.common_depth.data_transform,
        mask_background=config.common_depth.mask_background,
        normalize_data=config.common_depth.normalize_data,
        **config.common.__dict__,
    )
    data_spec = create_depth_dataspec_from_data_module(dm)
    return dm, data_spec


def get_depth_hp_data_module(config) -> Tuple[WoodscapeHPDepthDataModule, DepthDataSpec]:
    dm = hp_depth_datasets.WoodscapeHPDepthDataModule(
        pred_part=config.pred_part,
        nside=config.input_nside,
        base_pix=config.input_base_pix,
        shuffle_train_val_split=config.shuffle_train_val_split,
        data_transform=config.common_depth.data_transform,
        mask_background=config.common_depth.mask_background,
        normalize_data=config.common_depth.normalize_data,
        **config.common.__dict__,
    )
    data_spec = create_depth_dataspec_from_data_module(dm, base_pix=config.input_base_pix)
    return dm, data_spec


def get_data_module(data_config):
    data_dispatch = {
        WoodscapeFlatConfig.__name__: get_flat_data_module,
        WoodscapeHPConfig.__name__: get_hp_data_module,
        WoodscapeDepthFlatConfig.__name__: get_depth_flat_data_module,
        WoodscapeHPDepthConfig.__name__: get_depth_hp_data_module,
    }
    return data_dispatch[data_config.__class__.__name__](data_config)
