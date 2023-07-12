from typing import Union

import torch


class DataStats:
    def __init__(self, name, max, min, mean, std, total_pixels=None, total_background=None):
        self.name = name
        self.max = max
        self.min = min
        self.mean = mean
        self.std = std
        self.total_pixels = total_pixels
        self.total_background = total_background

    def __str__(self):
        s = ""
        s = s + "Data stats object:" + "\n"
        s = s + "Name: " + self.name + "\n"
        s = s + f"max: {self.max:.4f}" + "\n"
        s = s + f"min: {self.min:.4f}" + "\n"
        s = s + f"mean: {self.mean:.4f}" + "\n"
        s = s + f"std: {self.std:.4f}"
        if self.total_pixels is not None:
            s = s + "\n" + f"total_pixels: {self.total_pixels:.4f}"
        if self.total_background is not None:
            s = s + "\n" + f"total_background: {self.total_background:.4f}"
        return s


class MaskedDepthDataStatistics(DataStats):
    def __init__(self):
        super().__init__(
            name="Masked depth data stats",
            max=999.94287109375,
            min=0.16296708583831787,
            mean=13.654291032986958,
            std=29.58008801108711,
            total_pixels=2876849543,
        )


class DepthDataStatistics(DataStats):
    def __init__(self):
        super().__init__(
            name="Depth data stats",
            max=999.94287109375,
            min=0.16296708583831787,
            mean=53.27547067117465,
            std=195.83201099547819,
            total_pixels=2997248000,
            total_background=120398457,
        )


class LogDepthDataStatistics(DataStats):
    def __init__(self):
        super().__init__(
            name="Log depth data stats",
            max=6.907755374908447,
            min=-1.8142070770263672,
            mean=1.4544509182015166,
            std=2.0786484162088192,
        )


class MaskedLogDepthDataStatistics(DataStats):
    def __init__(self):
        super().__init__(
            name="Masked log depth data stats",
            max=6.907698154449463,
            min=-1.8142070770263672,
            mean=1.226225759977343,
            std=1.7902344298584563,
        )


class InvDepthDataStatistics(DataStats):
    def __init__(self):
        super().__init__(
            name="Inv depth data stats",
            max=6.136208534240723,
            min=0.001,
            mean=0.9910007833745446,
            std=1.449026079271616,
            total_pixels=2997248000,
            total_background=120398457,
        )


class MaskedInvDepthDataStatistics(DataStats):
    def __init__(self):
        super().__init__(
            name="Masked inv depth data stats",
            max=6.136208534240723,
            min=0.0010000570910051465,
            mean=1.0324331088958505,
            std=1.4645187100900352,
            total_pixels=2997248000,
            total_background=120398457,
        )


def get_depth_data_stats(
    data_transform: Union[None, str] = None, mask_background: bool = False
) -> Union[
    LogDepthDataStatistics,
    InvDepthDataStatistics,
    DepthDataStatistics,
    MaskedLogDepthDataStatistics,
    MaskedInvDepthDataStatistics,
    MaskedDepthDataStatistics,
]:
    if data_transform is None:
        data_transform = "None"

    data_stats = {
        False: {
            "log": LogDepthDataStatistics(),
            "inv": InvDepthDataStatistics(),
            "None": DepthDataStatistics(),
        },
        True: {
            "log": MaskedLogDepthDataStatistics(),
            "inv": MaskedInvDepthDataStatistics(),
            "None": MaskedDepthDataStatistics(),
        },
    }

    return data_stats[mask_background][data_transform]


def normalize_data(
    data: torch.Tensor, data_stats: DataStats, norm_type: Union[None, str] = None
) -> torch.Tensor:
    if norm_type is None or norm_type == "None":
        return data

    if norm_type == "standardize":
        data = (data - data_stats.mean) / data_stats.std
    elif norm_type == "min-max":
        data = (data - data_stats.min) / (data_stats.max - data_stats.min)

    return data


def unnormalize_data(
    data: torch.Tensor, data_stats: DataStats, norm_type: Union[None, str] = None
) -> torch.Tensor:
    if norm_type is None or norm_type == "None":
        return data

    if norm_type == "standardize":
        data = data * data_stats.std + data_stats.mean
    elif norm_type == "min-max":
        data = data * (data_stats.max - data_stats.min) + data_stats.min

    return data


def print_data_stats(data: torch.Tensor) -> None:

    inf_idxs = data.isinf()
    nan_idxs = data.isnan()
    idxs = ~(inf_idxs | nan_idxs)  # True = "foreground", false="background"

    print("Data stats:")
    print(f"max: {torch.max(data[idxs])  }")
    print(f"min: {torch.min(data[idxs])  }")
    print(f"mean: {torch.mean(data[idxs])}")
    print(f"std: {torch.std(data[idxs])  }")
    print(f"num infs: {torch.count_nonzero(inf_idxs)}")
    print(f"num nans: {torch.count_nonzero(nan_idxs)}")
