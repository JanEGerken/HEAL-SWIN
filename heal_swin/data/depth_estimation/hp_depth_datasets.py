import os
import argparse
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pytorch_lightning.core.datamodule import LightningDataModule
from healpy.pixelfunc import isnsideok

from heal_swin.utils import utils, get_paths, depth_utils
from heal_swin.data.depth_estimation import normalize_depth_data, flat_depth_datasets
from heal_swin.data.woodscape_dataset import WoodscapeDataset


class WoodscapeHPDepthImagesDataset(WoodscapeDataset):
    """Dataset of Woodscape Images on S2 in healpy format"""

    def __init__(
        self,
        nside=256,
        crop_green=False,
        cam_pos=None,
        train_share=0.8,
        shuffle_train_val_split=True,
        part="train",
        s2_bkgd_class=0,
        rotate_pole=False,
        base_pix=8,
        mask_background=False,
        data_transform=None,
        normalize_data=None,
    ):
        assert isnsideok(nside)
        assert 1 <= base_pix <= 12

        self.crop_green = crop_green
        self.nside = nside
        self.base_pix = base_pix

        self.normalize_data = normalize_data
        self.data_transform = data_transform
        self.mask_background = mask_background

        datasets_path = get_paths.get_syn_datasets_path()
        dataset_name = "hp_"
        dataset_name += f"depth_images_nside={self.nside}_base_pix={self.base_pix}"

        dataset_name += "_rotate_pole" if rotate_pole else ""
        self.root_dir = os.path.join(datasets_path, dataset_name)
        if not os.path.isdir(self.root_dir):
            print(
                f"Dataset of images and masks on S^2 in healpy grid not found at {self.root_dir},"
                + " generating dataset..."
            )
            from heal_swin.data.depth_estimation import project_depth_on_s2

            args_list = ["--nside", str(self.nside), "--samples", "-1"]
            args_list += ["--base_pix", str(self.base_pix)]
            args_list += ["--save_imgs_to", dataset_name]
            args_list += ["--part", "both"]
            args_list += ["--rotate_pole"] if rotate_pole else []
            args_list += ["--grid", "hp"]
            parser = argparse.ArgumentParser()
            parser = project_depth_on_s2.add_args(parser)
            args = parser.parse_args(args_list)
            project_depth_on_s2.main(args)
            utils.assert_path(self.root_dir)

        # call parent constructor only here because it uses get_dir, which returns self.root_dir
        super().__init__(
            cam_pos=cam_pos,
            train_share=train_share,
            part=part,
            shuffle_train_val_split=shuffle_train_val_split,
        )

        self.names = [os.path.splitext(file)[0] for file in self.file_names]

        self.data_stats = normalize_depth_data.get_depth_data_stats(
            data_transform=self.data_transform, mask_background=self.mask_background
        )

    def get_dir(self) -> str:
        return self.root_dir

    def get_extension(self) -> str:
        return ".npz"

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.paths[idx])

        hp_img = torch.tensor(data["hp_img"])
        hp_mask = torch.tensor(data["hp_mask"])

        hp_mask[hp_mask == 0] = float("inf")

        if self.mask_background:
            hp_mask[hp_mask == 1000] = float("inf")

        if self.data_transform:
            hp_mask = depth_utils.mask_transform_fcn(self.data_transform)(hp_mask)

        hp_mask = normalize_depth_data.normalize_data(
            data=hp_mask, data_stats=self.data_stats, norm_type=self.normalize_data
        )

        return hp_img, hp_mask

    def get_item_by_name(self, name) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self.names.index(name)
        return self[idx]


class WoodscapeHPDepthImagesPredictDataset(Dataset):
    """Dataset to generate predictions of Woodscape Depth masks on S2 in healpy grid from
    validation data masks

    This also returns the calibration data for projecting back onto the flat image and the original
    images and segmentation masks.
    """

    def __init__(
        self,
        nside=256,
        base_pix=8,
        crop_green=False,
        cam_pos=None,
        samples=40,
        base_train_share=0.8,
        shuffle_train_val_split=True,
        part="val",
        s2_bkgd_class=0,
        data_transform=None,
        normalize_data=None,
        rotate_pole=False,
        mask_background=False,
    ):
        assert isnsideok(nside)
        assert 1 <= base_pix <= 12

        super().__init__()

        dataset_kwargs = dict(
            nside=nside,
            base_pix=base_pix,
            crop_green=crop_green,
            cam_pos=cam_pos,
            s2_bkgd_class=s2_bkgd_class,
            rotate_pole=rotate_pole,
            shuffle_train_val_split=shuffle_train_val_split,
            mask_background=mask_background,
        )

        if isinstance(samples, float):  # samples is fraction of dataset to take
            assert 0.0 < samples <= 1.0
            if part == "train":
                train_share = base_train_share * samples
            elif part == "val":
                train_share = base_train_share + ((1 - samples) * (1 - base_train_share))
        elif isinstance(samples, int):
            dummy_ds = WoodscapeHPDepthImagesDataset(
                part="train",
                data_transform=data_transform,
                normalize_data=normalize_data,
                train_share=1.0,
                **dataset_kwargs,
            )
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

        self.hp_imgs_masks_dataset = WoodscapeHPDepthImagesDataset(
            part=part,
            train_share=train_share,
            data_transform=data_transform,
            normalize_data=normalize_data,
            **dataset_kwargs,
        )

        # take all the data for flat images and calibration, the matching is done in __getitem__
        self.imgs_masks_cal_dataset = flat_depth_datasets.WoodscapeDepthImagesCalibrationDataset(
            crop_green=crop_green,
            cam_pos=cam_pos,
            train_share=train_share,
            part=part,
            data_transform=data_transform,
            normalize_data=normalize_data,
            mask_background=mask_background,
        )

    def __len__(self) -> int:
        return len(self.hp_imgs_masks_dataset)

    def __getitem__(self, idx) -> dict:
        hp_img, hp_mask = self.hp_imgs_masks_dataset[idx]
        hp_name = os.path.splitext(self.hp_imgs_masks_dataset.file_names[idx])[0]
        img, mask, cal_info, name = self.imgs_masks_cal_dataset.get_item_by_name(hp_name)

        assert name == hp_name

        sample = {
            "hp_img": hp_img,
            "hp_mask": hp_mask,
            "img": img,
            "mask": mask,
            "cal_info": cal_info,
            "name": name,
        }

        return sample

    def collate_fn(self, data) -> dict:
        """collate_fn to aggregate samples into batches, to be used with PyTorch dataloaders"""

        batch = {}
        batch["hp_imgs"] = torch.stack([sample["hp_img"] for sample in data])
        batch["hp_masks"] = torch.stack([sample["hp_mask"] for sample in data])
        batch["imgs"] = torch.stack([sample["img"] for sample in data])
        batch["masks"] = torch.stack([sample["mask"] for sample in data])
        batch["cal_infos"] = [sample["cal_info"] for sample in data]
        batch["names"] = [sample["name"] for sample in data]
        return batch


class WoodscapeHPDepthDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size,
        val_batch_size,
        train_share,
        pred_part,
        train_worker,
        val_worker,
        manual_overfit_batches,
        shuffle,
        shuffle_train_val_split,
        cam_pos,
        nside,
        base_pix,
        pred_samples,
        seed,
        data_transform,
        normalize_data,
        rotate_pole,
        pred_batch_size,
        mask_background,
        *args,
        **kwargs,
    ):

        assert isnsideok(nside)
        assert 1 <= base_pix <= 12

        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.pred_batch_size = pred_batch_size
        self.shuffle = shuffle
        self.train_worker = train_worker
        self.val_worker = val_worker
        self.rotate_pole = rotate_pole
        self.nside = nside
        self.base_pix = base_pix
        self.pred_part = pred_part
        self.overfit_batches = manual_overfit_batches
        self.data_transform = data_transform
        self.normalize_data = normalize_data
        self.mask_background = mask_background

        dataset_kwargs = {
            "nside": nside,
            "base_pix": base_pix,
            "cam_pos": cam_pos,
            "train_share": train_share,
            "rotate_pole": rotate_pole,
            "shuffle_train_val_split": shuffle_train_val_split,
        }
        self.train_dataset = WoodscapeHPDepthImagesDataset(
            part="train",
            data_transform=data_transform,
            normalize_data=normalize_data,
            mask_background=mask_background,
            **dataset_kwargs,
        )
        self.val_dataset = WoodscapeHPDepthImagesDataset(
            part="val",
            data_transform=data_transform,
            normalize_data=normalize_data,
            mask_background=mask_background,
            **dataset_kwargs,
        )
        del dataset_kwargs["train_share"]
        pred_samples = -1 if self.overfit_batches > 0 else pred_samples
        self.pred_dataset = WoodscapeHPDepthImagesPredictDataset(
            samples=pred_samples,
            base_train_share=train_share,
            part=pred_part,
            data_transform=data_transform,
            normalize_data=normalize_data,
            mask_background=mask_background,
            **dataset_kwargs,
        )
        del dataset_kwargs["shuffle_train_val_split"]

        if self.overfit_batches > 0:
            samples = self.overfit_batches * self.batch_size
            # we need to seed the RNG since the datamodule gets instantiated several times for
            # training and prediction but we need the same training indices in both cases
            # this also means that the same seed will always select the same images
            rand_gen = torch.Generator()
            rand_gen.manual_seed(seed)
            self.train_idcs = torch.randperm(len(self.train_dataset), generator=rand_gen)[:samples]

    def get_train_overfit_names(self) -> List[str]:
        train_names = [self.train_dataset.file_names[idx] for idx in self.train_idcs]
        train_names.sort()
        return train_names

    def get_pred_overfit_sampler(self) -> SubsetRandomSampler:
        train_names = self.get_train_overfit_names()
        all_pred_names = self.pred_dataset.hp_imgs_masks_dataset.file_names
        pred_idcs = [np.where(all_pred_names == n)[0].item() for n in train_names]
        assert len(pred_idcs) == len(self.train_idcs)
        return SubsetRandomSampler(pred_idcs)

    def train_dataloader(self) -> DataLoader:
        if self.overfit_batches > 0:
            sampler = SubsetRandomSampler(self.train_idcs)
        else:
            sampler = None
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle if self.overfit_batches == 0 else False,
            num_workers=self.train_worker,
            sampler=sampler,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        if self.overfit_batches > 0:
            return self.train_dataloader()
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_worker,
        )
        return val_loader

    def predict_dataloader(self) -> DataLoader:
        if self.overfit_batches > 0:
            sampler = self.get_pred_overfit_sampler()
        else:
            sampler = None
        pred_loader = DataLoader(
            self.pred_dataset,
            batch_size=self.pred_batch_size,
            shuffle=False,
            num_workers=self.val_worker,
            collate_fn=self.pred_dataset.collate_fn,
            sampler=sampler,
        )
        return pred_loader

    def get_img_features(self) -> int:
        """Number of channels in pictures"""
        return 3

    def get_img_dims(self) -> Tuple[int, int]:
        img, mask = self.train_dataset[0]
        return img.shape[-1]

    def get_classes(self) -> int:
        """For compatibility, not relevant for depth regression"""
        return 1

    def get_original_img_dims(self) -> Tuple[int, int]:
        """Returns the dimensions of the original full-resolution images"""
        cal_info = self.pred_dataset[0]["cal_info"]
        height = cal_info["intrinsic"]["height"]
        width = cal_info["intrinsic"]["width"]
        return int(height), int(width)

    def get_pred_writer(
        self,
        pred_writer_name,
        output_dir,
        write_interval,
        output_resolution,
        prefix,
        top_k,
        ranking_metric,
        sort_dir,
        proj_res=1,
    ):
        from heal_swin.evaluation import hp_depth_pred_writers

        pred_writers = {
            None: hp_depth_pred_writers.WoodscapeHPDepthBasePredictionWriter,
            "base_writer": hp_depth_pred_writers.WoodscapeHPDepthBasePredictionWriter,
            "val_on_back_projected": (
                hp_depth_pred_writers.WoodscapeHPDepthValOnBackProjectedPredictionWriter
            ),
            "best_worst_preds": hp_depth_pred_writers.WoodscapeHPDepthBestWorstPredictionWriter,
            "chamfer_distance": (
                hp_depth_pred_writers.WoodscapeHPDepthChamferDistBestWorstPredictionWriter
            ),
        }
        kwargs = {
            "output_dir": output_dir,
            "write_interval": write_interval,
            "output_resolution": output_resolution,
            "rotate_pole": self.rotate_pole,
            "f_out": self.get_classes(),
            "prefix": prefix,
            "nside": self.nside,
            "base_pix": self.base_pix,
            "part": self.pred_part,
            "top_k": top_k,
            "ranking_metric": ranking_metric,
            "sort_dir": sort_dir,
            "pred_dataset": self.pred_dataset,
            "data_transform": self.data_transform,
            "mask_background": self.mask_background,
            "normalize_data": self.normalize_data,
            "img_dims": self.get_original_img_dims(),
        }
        assert (
            pred_writer_name in pred_writers.keys()
        ), f"prediction writer {pred_writer_name} unknow, implemented are {pred_writers.keys()}"

        return pred_writers[pred_writer_name](**kwargs)
