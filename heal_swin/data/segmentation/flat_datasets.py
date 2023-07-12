import os
import json

import numpy as np
from PIL import Image
import torch
import torchvision as tv
from torch.utils.data import Dataset

from heal_swin.utils import utils, get_paths
from heal_swin.data.woodscape_dataset import WoodscapeDataset


class WoodscapeImagesDataset(WoodscapeDataset):
    """Dataset of Woodscape RGB images"""

    def __init__(self, crop_green=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if crop_green:  # crop green strips
            self.transform = tv.transforms.CenterCrop((960, 1280))
        else:
            self.transform = utils.id

    def get_dir(self):
        datasets_path = get_paths.get_datasets_path(self.woodscape_version)
        return os.path.join(datasets_path, "rgb_images/rgb_images")

    def __getitem__(self, idx):
        img = tv.io.read_image(self.paths[idx])
        return self.transform(img)


class WoodscapeSemanticMasksDataset(WoodscapeDataset):
    """Dataset of Woodscape semantic segmentation masks"""

    def __init__(self, crop_green=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if crop_green:  # crop green strips
            self.transform = tv.transforms.CenterCrop((960, 1280))
        else:
            self.transform = utils.id

        self.names = [os.path.splitext(img_file)[0] for img_file in self.file_names]

    def get_dir(self):
        datasets_path = get_paths.get_datasets_path(self.woodscape_version)
        masks_path = os.path.join(
            datasets_path, "semantic_annotations/semantic_annotations/gtLabels"
        )
        return masks_path

    def __getitem__(self, idx):
        mask = Image.open(self.paths[idx])
        mask_torch = torch.as_tensor(np.array(mask))
        return self.transform(mask_torch)

    def get_item_by_name(self, name):
        """Get datapoint by name instead of index"""
        idx = self.names.index(name)
        return self[idx]


class WoodscapeCalibrationDataset(WoodscapeDataset):
    """Dataset of Woodscape calibration infos"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_extension(self):
        return ".json"

    def get_dir(self):
        datasets_path = get_paths.get_datasets_path(self.woodscape_version)
        cals_path = os.path.join(datasets_path, "calibration/calibration")
        return cals_path

    def __getitem__(self, idx):
        with open(self.paths[idx], "r") as f:
            cal_info = json.load(f)
        cal_info["file_name"] = self.file_names[idx]
        return cal_info


class WoodscapeSemanticImagesDataset(Dataset):
    """Dataset of Woodscape RGB images and semantic segmentation masks"""

    def __init__(self, crop_green=False, size=None, *args, **kwargs):
        super().__init__()

        self.imgs_dataset = WoodscapeImagesDataset(crop_green, *args, **kwargs)

        # set up masks dataset with the same files as images dataset
        self.masks_dataset = WoodscapeSemanticMasksDataset(crop_green, *args, **kwargs)
        masks_dir = self.masks_dataset.get_dir()
        self.masks_dataset.paths = [
            os.path.join(masks_dir, img) for img in self.imgs_dataset.file_names
        ]
        self.masks_dataset.update_paths()
        self.file_names = self.imgs_dataset.file_names

        self.img_transform = utils.id if size is None else tv.transforms.Resize(size)
        self.mask_transform = (
            utils.id if size is None else tv.transforms.Resize(size, interpolation=0)
        )
        self.padding = tv.transforms.Pad(kwargs.get("padding", [0, 0, 0, 0]))

    def __len__(self):
        return len(self.imgs_dataset)

    def __getitem__(self, idx):
        img = self.imgs_dataset[idx]
        mask = self.masks_dataset[idx]

        img_file = os.path.basename(self.imgs_dataset.file_names[idx])
        mask_file = os.path.basename(self.masks_dataset.file_names[idx])
        assert img_file == mask_file

        img = self.img_transform(img)
        img = self.padding(img)
        mask = self.mask_transform(mask[None, ...]).squeeze()
        mask = self.padding(mask)

        assert img.shape[-2] == mask.shape[-2]

        return img, mask


class WoodscapeSemanticImagesCalibrationDataset(Dataset):
    """Dataset of Woodscape RGB images, segmentation masks and calibration infos"""

    def __init__(self, crop_green=False, size=None, *args, **kwargs):
        super().__init__()
        self.imgs_dataset = WoodscapeImagesDataset(crop_green, *args, **kwargs)
        self.names = [os.path.splitext(img_file)[0] for img_file in self.imgs_dataset.file_names]
        self.file_names = self.imgs_dataset.file_names

        # set up calibration infos with the same files as images dataset
        self.cals_dataset = WoodscapeCalibrationDataset(*args, **kwargs)
        cals_dir = self.cals_dataset.get_dir()
        self.cals_dataset.paths = [
            os.path.join(cals_dir, img).replace(".png", ".json")
            for img in self.imgs_dataset.file_names
        ]
        self.cals_dataset.update_paths()

        # set up masks dataset with the same files as images dataset
        self.masks_dataset = WoodscapeSemanticMasksDataset(crop_green, *args, **kwargs)
        masks_dir = self.masks_dataset.get_dir()
        self.masks_dataset.paths = [
            os.path.join(masks_dir, img) for img in self.imgs_dataset.file_names
        ]
        self.masks_dataset.update_paths()

        self.img_transform = utils.id if size is None else tv.transforms.Resize(size)
        self.mask_transform = (
            utils.id if size is None else tv.transforms.Resize(size, interpolation=0)
        )
        self.padding = tv.transforms.Pad(kwargs.get("padding", [0, 0, 0, 0]))

    def __len__(self):
        return len(self.imgs_dataset)

    def __getitem__(self, idx):
        img = self.imgs_dataset[idx]
        mask = self.masks_dataset[idx]
        cal_info = self.cals_dataset[idx]

        img = self.img_transform(img)
        img = self.padding(img)
        mask = self.mask_transform(mask[None, ...]).squeeze()
        mask = self.padding(mask)

        img_file = os.path.basename(self.imgs_dataset.file_names[idx])
        mask_file = os.path.basename(self.masks_dataset.file_names[idx])
        cal_file = os.path.basename(self.cals_dataset.file_names[idx])
        assert img_file == cal_file.replace(".json", ".png") == mask_file

        name = self.names[idx]

        return img, mask, cal_info, name

    def get_item_by_name(self, name):
        """Get datapoint by name instead of index"""
        idx = self.names.index(name)
        return self[idx]
