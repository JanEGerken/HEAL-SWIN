import argparse
import os
import sys
import json
import pickle

from PIL import Image
import numpy as np
import torch
import torchvision as tv
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from heal_swin.data.segmentation import flat_datasets
from heal_swin.utils import utils, get_paths


def get_img_paths(woodscape_version):
    datasets_path = get_paths.get_datasets_path(woodscape_version)
    img_dir = os.path.join(datasets_path, "rgb_images/rgb_images")

    if not os.path.isdir(img_dir):
        print("Could not find rgb_images dir at", img_dir)
        sys.exit(1)

    png_in_img_dir = [entry.path for entry in os.scandir(img_dir) if ".png" in entry.name]
    fv = [fv_img for fv_img in png_in_img_dir if "FV" in fv_img]
    rv = [rv_img for rv_img in png_in_img_dir if "RV" in rv_img]
    mvl = [mvl_img for mvl_img in png_in_img_dir if "MVL" in mvl_img]
    mvr = [mvr_img for mvr_img in png_in_img_dir if "MVR" in mvr_img]

    fv.sort()
    rv.sort()
    mvl.sort()
    mvr.sort()

    result = {"fv": fv, "rv": rv, "mvl": mvl, "mvr": mvr}

    return result


def get_img_tensor(path):
    img = tv.io.read_image(path)
    return img


def get_img_shape_dtype(path):
    img = get_img_tensor(path)
    return img.shape, img.dtype


def get_label_tensor(path):
    img = Image.open(path)
    img_torch = torch.as_tensor(np.array(img))
    return img_torch


def get_label_shape_dtype(path):
    label = get_label_tensor(path)
    return label.shape, label.dtype


def get_sem_ann_path(woodscape_version):
    datasets_path = get_paths.get_datasets_path(woodscape_version)
    sem_ann_path = os.path.join(datasets_path, "semantic_annotations")
    return sem_ann_path


def get_sem_labels_paths(woodscape_version):
    labels_dir = os.path.join(
        get_sem_ann_path(woodscape_version), "semantic_annotations", "gtLabels"
    )

    if not os.path.isdir(labels_dir):
        print("Could not find gtLabels dir at", labels_dir)
        sys.exit(1)

    png_in_labels_dir = [entry.path for entry in os.scandir(labels_dir) if ".png" in entry.name]
    fv = [fv_label for fv_label in png_in_labels_dir if "FV" in fv_label]
    rv = [rv_label for rv_label in png_in_labels_dir if "RV" in rv_label]
    mvl = [mvl_label for mvl_label in png_in_labels_dir if "MVL" in mvl_label]
    mvr = [mvr_label for mvr_label in png_in_labels_dir if "MVR" in mvr_label]

    fv.sort()
    rv.sort()
    mvl.sort()
    mvr.sort()

    result = {"fv": fv, "rv": rv, "mvl": mvl, "mvr": mvr}

    return result


def get_seg_info(woodscape_version):
    seg_info_file = os.path.join(get_sem_ann_path(woodscape_version), "seg_annotation_info.json")

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


def gen_class_legend(args, img_paths, label_paths):
    seg_info = get_seg_info(args.woodscape_version)
    class_names = seg_info["class_names"]
    class_names = [f"{i}: {name}" for i, name in enumerate(class_names)]
    class_colors = seg_info["class_colors"]
    color_scheme = seg_info["color_scheme"] if "color_scheme" in seg_info.keys() else "RGB"
    cs_fct = get_color_scheme_fct(color_scheme)

    size = 1

    fig, ax = plt.subplots(1, 1)

    for idx in range(len(class_names)):
        x, y = 0, idx * 1.25 * size
        color = cs_fct(np.array(class_colors[idx])) / 255
        ax.add_patch(Rectangle((x, y), size, size, edgecolor="black", facecolor=color))
        ax.text(x + 1.25 * size, y + 0.5 * size, class_names[idx], va="center")

    ax.set_ylim(-0.25 * size, len(class_names) * 1.25 * size)
    ax.set_xlim(-0.25 * size, size + 5)
    ax.set_aspect(1)
    ax.set_axis_off()

    sem_ann_path = get_sem_ann_path(args.woodscape_version)
    file_path = os.path.join(sem_ann_path, "class_color_legend.pdf")
    fig.savefig(file_path, bbox_inches="tight", pad_inches=0)


def get_cal_distributions(pos, path, woodscape_version):
    def get_file_name(x):
        return os.path.splitext(os.path.basename(x))[0]

    cal_dataset = flat_datasets.WoodscapeCalibrationDataset(
        cam_pos=pos, train_share=1.0, part="train", woodscape_version=woodscape_version
    )
    data_names = [
        "aspect_ratio",
        "cx_offset",
        "cy_offset",
        "height",
        "k1",
        "k2",
        "k3",
        "k4",
        "poly_order",
        "width",
    ]
    cal_data_np = np.zeros((len(cal_dataset), len(data_names)), dtype=float)
    for idx, cal_info in enumerate(cal_dataset):
        int_cal_info = cal_info["intrinsic"]
        for i in range(len(data_names)):
            cal_data_np[idx, i] = int_cal_info[data_names[i]]
    unique, u_idcs, u_inverse, u_counts = np.unique(
        cal_data_np, axis=0, return_counts=True, return_index=True, return_inverse=True
    )
    print(
        f"Unique calibration data of {pos} dataset: there are {len(unique)} different calibrations:"
    )
    for i in range(len(unique)):
        for j in range(len(data_names)):
            print(f"{data_names[j]}: {unique[i,j]:.4f}")
        print(f"{u_counts[i]} samples with this calibration")
        print(
            f"first sample with this calibration: {os.path.basename(cal_dataset.paths[u_idcs[i]])}"
        )
        print("------")
    print("##############################################################")

    samples_by_cal_data = []
    for i in range(len(unique)):
        cal_info = cal_dataset[u_idcs[i]]
        del cal_info["file_name"]
        cal_data = {"cal_info": cal_info}
        cal_data["file_names"] = cal_dataset.paths[np.where(u_inverse == i)]
        cal_data["file_names"] = np.vectorize(get_file_name)(cal_data["file_names"])
        assert len(cal_data["file_names"]) == u_counts[i]
        samples_by_cal_data.append(cal_data)

    with open(path, "wb") as f:
        pickle.dump(samples_by_cal_data, f)


def gen_cal_distribution_data(args, img_paths, label_paths):
    file_names = []
    for pos in ["fv", "rv", "mvr", "mvl"]:
        path = os.path.join(
            get_paths.get_datasets_path(args.woodscape_version), f"{pos}_samples_by_cal_data.pickle"
        )
        file_names.append(path)

        if not os.path.isfile(path):
            get_cal_distributions(pos, path, args.woodscape_version)


def print_overview(args, img_paths, label_paths):
    total = 0
    for view, paths in img_paths.items():
        print(f"{len(paths)} images in {view} data")
        total += len(paths)
    print(f"In total: {total} images")

    shape, dtype = get_img_shape_dtype(img_paths["fv"][0])
    print(f"The images have shape {shape} and dtype {dtype}")

    total = 0
    for view, paths in label_paths.items():
        print(f"{len(paths)} labels in {view} data")
        total += len(paths)
    print(f"In total: {total} labels")

    shape, dtype = get_label_shape_dtype(label_paths["fv"][0])
    print(f"The labels have shape {shape} and dtype {dtype}")


TASK_DISPATCH = {
    "print_overview": print_overview,
    "gen_class_legend": gen_class_legend,
    "gen_cal_distribution_data": gen_cal_distribution_data,
}


def add_args(parent_parser):
    parent_parser.add_argument(
        "--woodscape_version", type=str, choices=utils.get_woodscape_versions()
    )
    parent_parser.add_argument("--part", type=str, choices=["train", "val", "both"])
    parent_parser.add_argument("--task", type=str, choices=TASK_DISPATCH.keys())
    return parent_parser


def main(args):

    img_paths = get_img_paths(args.woodscape_version)
    label_paths = get_sem_labels_paths(args.woodscape_version)

    TASK_DISPATCH[args.task](args, img_paths, label_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
    print("Done.")
