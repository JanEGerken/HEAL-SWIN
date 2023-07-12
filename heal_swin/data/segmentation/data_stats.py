import os
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np

from heal_swin.data.segmentation.flat_datasets import WoodscapeSemanticMasksDataset
from heal_swin.utils.get_paths import get_datasets_path
from heal_swin.utils import utils


def get_class_distribution(woodscape_version):
    ds = WoodscapeSemanticMasksDataset(part="train", woodscape_version=woodscape_version)
    dl = DataLoader(ds, batch_size=32)
    seg_info = utils.get_seg_info(woodscape_version)
    class_names = seg_info["class_names"]
    num_classes = len(class_names)

    class_counts = [0 for _ in range(num_classes)]
    pixel_count = 0
    for batch in tqdm(dl):
        for i in range(num_classes):
            class_counts[i] += (batch == i).sum()
        pixel_count += batch.numel()

    class_dist = [100 * class_count / pixel_count for class_count in class_counts]

    print(f"in total, there are {pixel_count} pixels in the train dataset")
    print("distribution of classes:")

    for i in range(num_classes):
        print(f"{i}\t{class_names[i]: <20}\t{class_dist[i]:.3f}%")

    return class_dist


def plot_hist(class_freqs, file_name, woodscape_version):
    n_classes = len(class_freqs)
    seg_info = utils.get_seg_info(woodscape_version)
    class_names = seg_info["class_names"]
    idcs = np.flip(np.argsort(class_freqs))
    class_freqs = np.array(class_freqs)[idcs]
    class_names = np.array(class_names)[idcs]

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(15, 5)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=5,
            )

    width = 0.8
    bars = ax.bar(np.arange(n_classes) - width / 2 + width / 2, class_freqs, width)
    autolabel(bars)

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation="vertical")
    ax.set_ylabel("percent of total pixels in subset")

    for item in [ax.title, ax.xaxis.label] + ax.get_xticklabels():
        item.set_fontsize(8)

    datasets_path = get_datasets_path(woodscape_version)
    save_path = os.path.join(datasets_path, "semantic_annotations", file_name)
    print(f"saving histogram at {save_path}")
    fig.savefig(save_path, bbox_inches="tight")


def main(args):
    class_dist = get_class_distribution(args.woodscape_version)

    plot_hist(class_dist, "class_hist.pdf", args.woodscape_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--woodscape_version", type=str, choices=utils.get_woodscape_versions())
    args = parser.parse_args()
    main(args)
    print("Done.")
