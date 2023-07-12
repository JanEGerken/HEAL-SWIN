from typing import Tuple, Union, Optional, List
from dataclasses import dataclass


@dataclass
class DataSpec:
    dim_in: Union[int, Tuple[int, int]]  # single int for healpy
    f_in: int
    f_out: int
    base_pix: Optional[int]
    class_names: List[str]


def create_dataspec_from_data_module(dm, base_pix=0):
    f_in = dm.get_img_features() if dm.get_img_features() > 2 else 1
    f_out = dm.get_classes()
    input_img_dims = dm.get_img_dims()
    class_names = dm.get_class_names()

    return DataSpec(
        f_in=f_in, f_out=f_out, dim_in=input_img_dims, base_pix=base_pix, class_names=class_names
    )
