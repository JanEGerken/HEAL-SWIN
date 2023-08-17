import os
from typing import Tuple
import json

import numpy as np
import torch
import torchvision as tv
from torch.utils.data import Dataset

from heal_swin.utils import utils
from heal_swin.utils import get_paths
from heal_swin.data.depth_estimation import normalize_depth_data
from heal_swin.utils import depth_utils
from heal_swin.data.depth_estimation import hp_depth_datasets
from heal_swin.data.woodscape_dataset import WoodscapeDataset


def change_extension(string, extention) -> str:
    return ".".join([string.split(".")[0], extention])


def check_same_base_filename(name1, name2) -> bool:
    return name1.split(".")[0] == name2.split(".")[0]


class WoodscapeDepthImagesDataset(WoodscapeDataset):
    """Dataset of Syn-Woodscape RGB images"""

    def __init__(self, crop_green=False, size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = utils.id if size is None else tv.transforms.Resize(size)

    def get_dir(self) -> str:
        syn_path = get_paths.get_syn_datasets_path()
        return os.path.join(syn_path, "rgb_images/rgb_images")

    def get_extension(self) -> str:
        return ".png"

    def __getitem__(self, idx) -> torch.Tensor:
        img = tv.io.read_image(self.paths[idx])
        return self.transform(img)


class WoodscapeDepthMaskDataset(WoodscapeDataset):
    """Dataset of Woodscape RGB images"""

    def __init__(
        self,
        crop_green=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def get_dir(self) -> str:
        syn_path = get_paths.get_syn_datasets_path()
        return os.path.join(syn_path, "depth_maps/raw_data")

    def get_extension(self) -> str:
        return ".npy"

    def __getitem__(self, idx) -> torch.Tensor:
        mask = torch.tensor(np.load(self.paths[idx]), dtype=torch.float32)

        return mask


class WoodscapeDepthDataset(Dataset):
    def __init__(
        self,
        size=None,
        crop_green=False,
        data_transform=None,
        mask_background=False,
        normalize_data=None,
        interpolation_mode ="nearest",
        *args,
        **kwargs,
    ):
        self.imgs_dataset = WoodscapeDepthImagesDataset(*args, **kwargs)
        self.depth_masks_dataset = WoodscapeDepthMaskDataset(
            *args,
            **kwargs,
        )

        assert len(self.imgs_dataset) == len(self.depth_masks_dataset)

        depth_masks_dir = self.depth_masks_dataset.get_dir()
        self.depth_masks_dataset.paths = [
            os.path.join(depth_masks_dir, change_extension(img, "npy"))
            for img in self.imgs_dataset.file_names
        ]  # Make sure that the image and mask have the same base name
        self.depth_masks_dataset.update_paths()
        self.file_names = self.imgs_dataset.file_names

        self.img_transform = utils.id if size is None else tv.transforms.Resize(size)
        self.interpolation_modes = {
            "nearest": tv.transforms.InterpolationMode.NEAREST,
            "bilinear":tv.transforms.InterpolationMode.BILINEAR
        }
        self.interpolation_mode = interpolation_mode
        self.mask_transform = (
            utils.id
            if size is None
            else tv.transforms.Resize(size, interpolation=self.interpolation_modes[self.interpolation_mode])
        )
        self.padding = tv.transforms.Pad(kwargs.get("padding", [0, 0, 0, 0]))

        self.mask_background = mask_background
        self.data_transform = data_transform
        self.normalize_data = normalize_data
        self.data_stats = normalize_depth_data.get_depth_data_stats(
            data_transform=self.data_transform, mask_background=self.mask_background
        )

    def __len__(self) -> int:
        return len(self.imgs_dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.imgs_dataset[idx]
        mask = self.depth_masks_dataset[idx]

        img_file = os.path.basename(self.imgs_dataset.file_names[idx])
        mask_file = os.path.basename(self.depth_masks_dataset.file_names[idx])
        assert check_same_base_filename(img_file, mask_file)

        img = self.img_transform(img)
        img = self.padding(img)
        mask = self.mask_transform(mask[None, ...]).squeeze()
        mask = self.padding(mask)

        assert img.shape[-2] == mask.shape[-2]

        if self.mask_background:
            mask[mask == 1000] = float("inf")

        if self.data_transform:
            mask = depth_utils.mask_transform_fcn(self.data_transform)(mask)

        mask = normalize_depth_data.normalize_data(
            data=mask, data_stats=self.data_stats, norm_type=self.normalize_data
        )

        return img, mask


class WoodscapeDepthCalibrationDataset(WoodscapeDataset):
    """Dataset of Woodscape calibration infos"""

    def __init__(self, crop_green=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_extension(self) -> str:
        return ".json"

    def get_dir(self) -> str:
        syn_path = get_paths.get_syn_datasets_path()

        cals_path = os.path.join(syn_path, "calibration/calibration")
        return cals_path

    def __getitem__(self, idx) -> dict:
        with open(self.paths[idx], "r") as f:
            cal_info = json.load(f)
        cal_info["file_name"] = self.file_names[idx]
        return cal_info


class WoodscapeDepthImagesCalibrationDataset(Dataset):

    """Dataset of Woodscape RGB images, segmentation masks and calibration infos"""

    def __init__(
        self,
        size=None,
        interpolation_mode="nearest",
        data_transform=None,
        mask_background=False,
        normalize_data=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.imgs_dataset = WoodscapeDepthImagesDataset(*args, **kwargs)
        self.names = [os.path.splitext(img_file)[0] for img_file in self.imgs_dataset.file_names]
        self.file_names = self.imgs_dataset.file_names

        # set up calibration infos with the same files as images dataset
        self.cals_dataset = WoodscapeDepthCalibrationDataset(*args, **kwargs)

        cals_dir = self.cals_dataset.get_dir()
        self.cals_dataset.paths = [
            os.path.join(cals_dir, img).replace(".png", ".json")
            for img in self.imgs_dataset.file_names
        ]
        self.cals_dataset.update_paths()

        # set up masks dataset with the same files as images dataset
        self.masks_dataset = WoodscapeDepthMaskDataset(
            *args,
            **kwargs,
        )
        masks_dir = self.masks_dataset.get_dir()
        self.masks_dataset.paths = [
            os.path.join(masks_dir, img.replace(".png", ".npy"))
            for img in self.imgs_dataset.file_names
        ]
        self.masks_dataset.update_paths()

        self.img_transform = utils.id if size is None else tv.transforms.Resize(size)
        self.interpolation_modes = {
            "nearest": tv.transforms.InterpolationMode.NEAREST,
            "bilinear":tv.transforms.InterpolationMode.BILINEAR
        }
        self.interpolation_mode = interpolation_mode
        self.mask_transform = (
            utils.id
            if size is None
            else tv.transforms.Resize(size, interpolation=self.interpolation_modes[self.interpolation_mode])
        )
        self.padding = tv.transforms.Pad(kwargs.get("padding", [0, 0, 0, 0]))
        self.mask_background = mask_background

    def __len__(self) -> int:
        return len(self.imgs_dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, dict, str]:
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
        assert img_file == cal_file.replace(".json", ".png") == mask_file.replace(".npy", ".png"), (
            "The files should have the same base name got img_file: "
            f"{img_file}, mask_file: {mask_file}, cal_file: {cal_file}"
        )

        name = self.names[idx]

        return img, mask, cal_info, name

    def get_item_by_name(self, name) -> Tuple[torch.Tensor, torch.Tensor, dict, str]:
        """Get datapoint by name instead of index"""
        idx = self.names.index(name)
        return self[idx]


class WoodscapeDepthImagesPredictDataset(Dataset):
    """Dataset to generate predictions of Woodscape Semantic masks in flat CNN from validation masks

    This also returns the original images and segmentation masks.
    """

    def __init__(
        self,
        crop_green=False,
        cam_pos=None,
        samples=40,
        base_train_share=0.8,
        part="val",
        size=None,
        padding=[0, 0, 0, 0],
        shuffle_train_val_split=True,
        nside=256,
        base_pix=8,
        s2_bkgd_class=0,
        rotate_pole=False,
        data_transform=None,
        mask_background=False,
        normalize_data=None,
    ):
        super().__init__()

        dataset_kwargs = {
            "crop_green": crop_green,
            "cam_pos": cam_pos,
            "size": size,
            "padding": padding,
            "shuffle_train_val_split": shuffle_train_val_split,
        }

        if isinstance(samples, float):  # samples is fraction of dataset to take
            assert 0.0 < samples <= 1.0
            if part == "train":
                train_share = base_train_share * samples
            elif part == "val":
                train_share = base_train_share + ((1 - samples) * (1 - base_train_share))
        elif isinstance(samples, int):
            dummy_ds = WoodscapeDepthDataset(part="train", train_share=1.0, **dataset_kwargs)
            total_imgs = len(dummy_ds)
            assert (
                samples <= total_imgs
            ), f"requested {samples} predict samples, but dataset has only {total_imgs} images"
            if part == "train":
                train_share = samples / total_imgs
            elif part == "val":
                # set train_share so that validation data are last 'samples' samples
                train_share = 1 - samples / total_imgs
            if samples == -1:  # special case for overfitting on a few samples: select everything
                train_share = 1.0
                part = "train"

        dataset_kwargs["train_share"] = train_share
        dataset_kwargs["part"] = part

        self.transf_sem_img_dataset = WoodscapeDepthDataset(
            data_transform=data_transform,
            mask_background=mask_background,
            normalize_data=normalize_data,
            **dataset_kwargs,
        )
        dataset_kwargs["size"] = None  # unresized images

        self.sem_img_dataset = WoodscapeDepthImagesCalibrationDataset(
            data_transform=data_transform,
            mask_background=mask_background,
            normalize_data=normalize_data,
            **dataset_kwargs,
        )

        # take all the data for flat images and calibration, the matching is done in __getitem__
        self.hp_dataset = hp_depth_datasets.WoodscapeHPDepthImagesDataset(
            crop_green=crop_green,
            cam_pos=cam_pos,
            train_share=train_share,
            part=part,
            nside=nside,
            base_pix=base_pix,
            s2_bkgd_class=s2_bkgd_class,
            rotate_pole=rotate_pole,
            data_transform=data_transform,
            mask_background=mask_background,
            normalize_data=normalize_data,
        )

    def __len__(self) -> int:
        return len(self.sem_img_dataset)

    def __getitem__(self, idx) -> dict:
        transf_img, transf_mask = self.transf_sem_img_dataset[idx]
        img, mask, cal_info, name = self.sem_img_dataset[idx]

        assert name == os.path.splitext(self.transf_sem_img_dataset.file_names[idx])[0]

        hp_img, hp_mask = self.hp_dataset.get_item_by_name(name)

        sample = {
            "s2_img": transf_img,
            "s2_mask": transf_mask,
            "img": img,
            "mask": mask,
            "cal_info": cal_info,
            "hp_img": hp_img,
            "hp_mask": hp_mask,
            "name": name,
        }

        return sample

    def collate_fn(self, data) -> dict:
        """collate_fn to aggregate samples into batches, to be used with PyTorch dataloaders"""

        batch = {}
        batch["hp_imgs"] = torch.stack([sample["hp_img"] for sample in data])
        batch["hp_masks"] = torch.stack([sample["hp_mask"] for sample in data])
        batch["s2_imgs"] = torch.stack([sample["s2_img"] for sample in data])
        batch["s2_masks"] = torch.stack([sample["s2_mask"] for sample in data])
        batch["imgs"] = torch.stack([sample["img"] for sample in data])
        batch["masks"] = torch.stack([sample["mask"] for sample in data])
        batch["cal_infos"] = [sample["cal_info"] for sample in data]
        batch["names"] = [sample["name"] for sample in data]
        return batch
