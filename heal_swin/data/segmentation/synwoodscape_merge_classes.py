import os
import json

from tqdm import tqdm
from PIL import Image
from flat_datasets import WoodscapeSemanticMasksDataset

from heal_swin.utils import utils, get_paths

# labels of all synthetic woodscape classes in new dataset
CLASS_MAPPING = [0, 1, 0, 0, 0, 0, 2, 3, 4, 0, 5, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7]
NEW_CLASS_NAMES = [
    "void",
    "building",
    "road line",
    "road",
    "sidewalk",
    "four-wheeler vehicle",
    "sky",
    "ego-vehicle",
]

DATASET_NAME = "synwoodscape_large"


def main():
    """This function generates a new version of the SynWoodScapes dataset out of the original data
    by merging classes.

    The constant CLASS_MAPPING is expected to contain for each class of the original dataset, the
    index of that class in the new dataset.
    The constant NEW_CLASS_NAMES is expected to contain the names of the new classes.
    The constant DATASET_NAME is expected to contain the name of the new dataset.
    """
    ds = WoodscapeSemanticMasksDataset(
        part="train", train_share=1.0, woodscape_version="synwoodscape"
    )

    old_ds_root_path = get_paths.get_datasets_path("synwoodscape")
    ds_root_path = os.path.abspath(os.path.join(old_ds_root_path, "../"))
    new_ds_root_path = os.path.join(ds_root_path, DATASET_NAME)

    os.makedirs(new_ds_root_path)
    os.makedirs(os.path.join(new_ds_root_path, "calibration"))
    os.makedirs(os.path.join(new_ds_root_path, "calibration(test_set)"))
    os.makedirs(os.path.join(new_ds_root_path, "rgb_images"))
    os.makedirs(os.path.join(new_ds_root_path, "rgb_images(test_set)"))

    label_dir = os.path.join(new_ds_root_path, "semantic_annotations/semantic_annotations/gtLabels")
    os.makedirs(label_dir)

    os.symlink(
        os.path.join(old_ds_root_path, "calibration", "calibration"),
        os.path.join(new_ds_root_path, "calibration", "calibration"),
    )
    os.symlink(
        os.path.join(old_ds_root_path, "calibration(test_set)", "calibration(test_set)"),
        os.path.join(new_ds_root_path, "calibration(test_set)", "calibration(test_set)"),
    )
    os.symlink(
        os.path.join(old_ds_root_path, "rgb_images", "rgb_images"),
        os.path.join(new_ds_root_path, "rgb_images", "rgb_images"),
    )
    os.symlink(
        os.path.join(old_ds_root_path, "rgb_images(test_set)", "rgb_images(test_set)"),
        os.path.join(new_ds_root_path, "rgb_images(test_set)", "rgb_images(test_set)"),
    )

    for idx, mask in tqdm(enumerate(ds), total=len(ds)):
        new_mask = mask.clone()
        for i in range(len(CLASS_MAPPING)):
            new_mask[mask == i] = CLASS_MAPPING[i]
        name = ds.file_names[idx]
        path = os.path.join(label_dir, name)
        Image.fromarray(new_mask.numpy()).save(path)

    num_new_classes = len(NEW_CLASS_NAMES)
    seg_info = utils.get_seg_info("synwoodscape")
    seg_info["class_names"] = NEW_CLASS_NAMES
    seg_info["class_colors"] = seg_info["class_colors"][:num_new_classes]
    seg_info["class_indexes"] = list(range(num_new_classes))

    seg_info_path = os.path.join(new_ds_root_path, "semantic_annotations/seg_annotation_info.json")
    with open(seg_info_path, "w") as f:
        json.dump(seg_info, f, indent=2)


if __name__ == "__main__":
    main()
    print("Done.")
