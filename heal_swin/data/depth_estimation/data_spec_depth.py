from dataclasses import dataclass
from typing import Optional, Union, Tuple

from heal_swin.data.depth_estimation.flat_depth_datamodule import WoodscapeFlatDepthDataModule
from heal_swin.data.depth_estimation.hp_depth_datasets import WoodscapeHPDepthDataModule
from heal_swin.data.depth_estimation.normalize_depth_data import (
    MaskedDepthDataStatistics,
    DepthDataStatistics,
    LogDepthDataStatistics,
    MaskedLogDepthDataStatistics,
    InvDepthDataStatistics,
    MaskedInvDepthDataStatistics,
    get_depth_data_stats,
)


@dataclass
class DepthDataSpec:
    dim_in: Union[int, Tuple[int, int]]  # single int for healpy
    f_in: int
    f_out: int
    base_pix: Optional[int]
    data_stats: Union[
        MaskedDepthDataStatistics,
        DepthDataStatistics,
        LogDepthDataStatistics,
        MaskedLogDepthDataStatistics,
        InvDepthDataStatistics,
        MaskedInvDepthDataStatistics,
    ]


def create_depth_dataspec_from_data_module(
    dm: Union[WoodscapeFlatDepthDataModule, WoodscapeHPDepthDataModule],
    base_pix: int = 0,
) -> DepthDataSpec:
    f_in = dm.get_img_features() if dm.get_img_features() > 2 else 1
    f_out = dm.get_classes()
    input_img_dims = dm.get_img_dims()

    data_transform = dm.data_transform
    mask_background = dm.mask_background

    data_stats = get_depth_data_stats(
        data_transform=data_transform,
        mask_background=mask_background,
    )

    return DepthDataSpec(
        f_in=f_in, f_out=f_out, dim_in=input_img_dims, base_pix=base_pix, data_stats=data_stats
    )
