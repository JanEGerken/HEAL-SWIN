import os
import sys
import json
import re
import importlib

import torch
import numpy as np
import pandas as pd

from heal_swin.utils.get_paths import (
    get_datasets_path,
    get_mlruns_path,
    get_abs_path_from_config_path,
)
from heal_swin.utils import serialize


def id(x):
    """identity function (to avoid lambda functions for pickle)"""
    return x


def nanmean(tensor):
    """returns mean of all non-nan elements of torch tensor"""
    return torch.mean(tensor[~torch.isnan(tensor)])


def assert_path(path):
    if not os.path.isdir(path):
        print("Could not find directory at", path, file=sys.stderr)
        sys.exit(1)


def get_seg_info(woodscape_version):
    datasets_path = get_datasets_path(woodscape_version)
    seg_info_file = os.path.join(datasets_path, "semantic_annotations", "seg_annotation_info.json")

    if not os.path.isfile(seg_info_file):
        print("Could not find seg_annotation_info.json file at", seg_info_file)
        sys.exit(1)

    with open(seg_info_file, "r") as f:
        seg_info = json.load(f)

    return seg_info


def get_color_scheme_fct(color_scheme):
    if color_scheme == "RGB":
        return lambda x: x

    elif color_scheme == "BGR":

        def general_flip(x):
            if isinstance(x, torch.Tensor):
                return torch.flip(x, (0,))
            elif isinstance(x, np.ndarray):
                return np.flip(x)
            else:
                print(f"unknown type for color scheme conversion: {type(x)}")
                sys.exit(1)

        return general_flip
    else:
        print(f"unknown color scheme: {color_scheme}")
        sys.exit(1)


def gen_mask_img(seg_info, cs_fct, mask, color_info=None):
    """Generate colored image out of segmentation mask using colors from segmentation infos"""

    if color_info is None:
        color_info = (seg_info["class_indexes"], seg_info["class_colors"], cs_fct)

    class_indices, class_colors, cs_fct = color_info
    mask_img = torch.zeros((3,) + tuple(mask.shape), dtype=torch.uint8)

    for value, color in zip(class_indices, class_colors):
        mask_img += (mask == value) * np.reshape(
            cs_fct(torch.tensor(color)), (3,) + (1,) * mask.ndim
        )

    return mask_img


def overlay_from_seg_info(mask, img, seg_info, cs_fct=id, mask_opacity=0.4):
    color_info = (seg_info["class_indexes"], seg_info["class_colors"], cs_fct)

    if isinstance(mask, np.ndarray):
        mask = torch.tensor(mask)
    if isinstance(img, np.ndarray):
        img = torch.tensor(img)

    mask_img = gen_mask_img(seg_info, cs_fct, mask, color_info)
    overlay = (mask_img != 0) * ((1.0 - mask_opacity) * img + mask_opacity * mask_img)
    overlay += (mask_img == 0) * img
    return overlay.byte()


def get_overlay(woodscape_version, mask, img, mask_opacity=0.4):
    """Combine mask and image into an overlay"""
    seg_info = get_seg_info(woodscape_version)
    color_scheme = seg_info["color_scheme"] if "color_scheme" in seg_info.keys() else "RGB"
    cs_fct = get_color_scheme_fct(color_scheme)
    return overlay_from_seg_info(mask, img, seg_info, cs_fct, mask_opacity)


def flatten_dict(nested_dict, sep="_"):
    df = pd.json_normalize(nested_dict, sep=sep)
    return df.to_dict(orient="records")[0]


def get_woodscape_versions():
    woodscape_path = get_datasets_path("woodscape")
    woodscape_root = os.path.abspath(os.path.join(woodscape_path, ".."))
    versions = []
    for entry in os.scandir(woodscape_root):
        if os.path.isdir(entry.path):
            versions.append(entry.name)
    return versions


def get_artifacts_path(run_id):
    mlruns_path = get_mlruns_path()  # "mlruns"
    found_expmt = None
    for expmt in list(os.listdir(mlruns_path)):
        if os.path.isdir(os.path.join(mlruns_path, expmt)):
            if run_id in os.listdir(os.path.join(mlruns_path, expmt)):
                found_expmt = expmt
                break

    if found_expmt is None:
        raise RuntimeError(f"The specified run_id {run_id} wasn't found")

    artifact_path = os.path.join(mlruns_path, found_expmt, run_id, "artifacts")
    assert os.path.isdir(artifact_path), f"{artifact_path} is not a directory"
    return artifact_path


def check_and_get_ckpt_paths(run_identifier, epoch, epoch_number=None):
    """From a run identifier (full path or mlflow run_id) this
    extracts the artifacts folder and the ckpt that is to be
    used for evaluation (based on the options submitted)."""

    # Check if run_identifier is ok if so return path
    explicit_path = True
    if os.path.isfile(run_identifier) and run_identifier.split(".")[-1] == "ckpt":
        print(f"Found matching ckpt {run_identifier}")
        return (
            run_identifier,
            "/".join(run_identifier.split("/")[:-1]),
            explicit_path,
        )  # ckpt path and artifact path

    explicit_path = False

    if isinstance(epoch, int):
        epoch = str(epoch)

    artifact_path = get_artifacts_path(run_identifier)

    ckpts = []
    artifacts = os.listdir(artifact_path)
    for art in artifacts:
        if art[-4:] == "ckpt" in art:
            ckpts.append(art)

    if epoch.lower() == "last":
        assert os.path.isfile(os.path.join(artifact_path, "last.ckpt")), (
            f"For run {run_identifier} there is no explicit last ckpt, "
            f"the available ckpts are {ckpts}"
        )
        ckpt = "last.ckpt"

    elif epoch.lower() == "best":
        assert os.path.isfile(os.path.join(artifact_path, "best.ckpt")), (
            f"For run {run_identifier} there is no explicit best ckpt, "
            f"the available ckpts are {ckpts}"
        )
        ckpt = "best.ckpt"

    elif epoch == "number":
        assert epoch_number.isdigit(), (
            f"You need to speficy which epoch number to load, " f"currently given {epoch_number}"
        )
        r = re.compile(f"epoch={epoch_number}_.+\\.ckpt")
        matches = list(filter(r.match, ckpts))
        assert (
            len(matches) == 1
        ), f"did not find exactly one matching checkpoint, found: {matches} among {ckpts}"
        ckpt = matches[0]
    else:
        raise NotImplementedError(f"The specified eval epoch {epoch} was not implemented")

    ckpt_path = os.path.join(artifact_path, ckpt)

    return ckpt_path, artifact_path, explicit_path


def load_config(run_id, config_name):
    artifact_path = get_artifacts_path(run_id)
    config_path = os.path.join(artifact_path, config_name)

    assert os.path.isfile(config_path), f"config at {config_path} not found"
    return serialize.load(config_path)


def get_config_from_config_path(config_path, get_fct_name):
    abs_path = get_abs_path_from_config_path(config_path)
    assert os.path.isfile(abs_path), f"configuration at {abs_path} not found"
    loader = importlib.machinery.SourceFileLoader("config", abs_path)
    spec = importlib.util.spec_from_loader("config", loader)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return getattr(config, get_fct_name)()
