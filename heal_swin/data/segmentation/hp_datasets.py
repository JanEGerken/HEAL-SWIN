import os
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pytorch_lightning.core.datamodule import LightningDataModule
from healpy.pixelfunc import isnsideok

from heal_swin.utils import utils
from heal_swin.utils import get_paths
from heal_swin.data.woodscape_dataset import WoodscapeDataset
from heal_swin.data.segmentation import flat_datasets
from heal_swin.evaluation import hp_pred_writers
from heal_swin.evaluation.hp_pred_writers import (
    WoodscapeHPBackProjectedHPMaskedIoUPredictionWriter,
)


class WoodscapeHPSemanticImagesDataset(WoodscapeDataset):
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
        woodscape_version=None,
        training_data_fraction=1.0,
        data_fraction_seed=42,
    ):
        assert isnsideok(nside)
        assert 1 <= base_pix <= 12

        self.crop_green = crop_green
        self.nside = nside
        self.base_pix = base_pix

        datasets_path = get_paths.get_datasets_path(woodscape_version)
        dataset_name = f"hp_images_nside={self.nside}_base_pix={self.base_pix}"
        dataset_name += f"_crop_green={self.crop_green}"
        # for backwards compatibility keep the background class out of the name if it is 0:
        dataset_name += f"_s2_bkgd_class={s2_bkgd_class}" if s2_bkgd_class != 0 else ""
        # same for rotate_pole:
        dataset_name += "_rotate_pole" if rotate_pole else ""
        self.root_dir = os.path.join(datasets_path, dataset_name)
        if not os.path.isdir(self.root_dir):
            print(
                f"Dataset of images and masks on S^2 in HEALPix grid not found at {self.root_dir},"
                + " generating dataset..."
            )
            from heal_swin.data.segmentation import project_on_s2

            args_list = ["--nside", str(self.nside), "--samples", "-1"]
            args_list += ["--base_pix", str(self.base_pix)]
            args_list += ["--save_imgs_to", dataset_name]
            args_list += ["--part", "both"]
            args_list += ["--s2_bkgd_class", str(s2_bkgd_class)]
            args_list += ["--rotate_pole"] if rotate_pole else []
            args_list += ["--woodscape_version", woodscape_version]
            parser = argparse.ArgumentParser()
            parser = project_on_s2.add_args(parser)
            args = parser.parse_args(args_list)
            project_on_s2.main(args)
            utils.assert_path(self.root_dir)

        # call parent constructor only here because it uses get_dir, which returns self.root_dir
        super().__init__(
            cam_pos=cam_pos,
            train_share=train_share,
            part=part,
            shuffle_train_val_split=shuffle_train_val_split,
            woodscape_version=woodscape_version,
            training_data_fraction=training_data_fraction,
            data_fraction_seed=data_fraction_seed,
        )

        self.names = [os.path.splitext(file)[0] for file in self.file_names]

    def get_dir(self):
        return self.root_dir

    def get_extension(self):
        return ".npz"

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])

        hp_img = data["hp_img"]
        hp_mask = data["hp_mask"]

        return hp_img, hp_mask

    def get_item_by_name(self, name):
        idx = self.names.index(name)
        return self[idx]


class WoodscapeHPSemanticImagesPredictDataset(Dataset):
    """Dataset to generate predictions of Woodscape Semantic masks on S2 in healpy grid from
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
        rotate_pole=False,
        woodscape_version=None,
        training_data_fraction=1.0,
        data_fraction_seed=42,
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
            woodscape_version=woodscape_version,
            training_data_fraction=training_data_fraction,
            data_fraction_seed=data_fraction_seed,
        )

        if isinstance(samples, float):  # samples is fraction of dataset to take
            assert 0.0 < samples <= 1.0
            if part == "train":
                train_share = base_train_share * samples
            elif part == "val":
                train_share = base_train_share + (
                    (1 - samples) * (1 - base_train_share)
                )
        elif isinstance(samples, int):
            dummy_ds = WoodscapeHPSemanticImagesDataset(
                part="train", train_share=1.0, **dataset_kwargs
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
            if (
                samples == -1
            ):  # special case for overfitting on a few samples: select everything
                train_share = 1.0
                part = "train"

        self.hp_imgs_masks_dataset = WoodscapeHPSemanticImagesDataset(
            part=part, train_share=train_share, **dataset_kwargs
        )
        # take all the data for flat images and calibration, the matching is done in __getitem__
        self.imgs_masks_cal_dataset = (
            flat_datasets.WoodscapeSemanticImagesCalibrationDataset(
                crop_green=crop_green,
                cam_pos=cam_pos,
                train_share=0,
                part="val",
                woodscape_version=woodscape_version,
            )
        )

    def __len__(self):
        return len(self.hp_imgs_masks_dataset)

    def __getitem__(self, idx):
        hp_img, hp_mask = self.hp_imgs_masks_dataset[idx]
        hp_name = os.path.splitext(self.hp_imgs_masks_dataset.file_names[idx])[0]
        img, mask, cal_info, name = self.imgs_masks_cal_dataset.get_item_by_name(
            hp_name
        )

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

    def get_item_by_name(self, name):
        idx = self.hp_imgs_masks_dataset.names.index(name)
        return self[idx]

    def collate_fn(self, data):
        """collate_fn to aggregate samples into batches, to be used with PyTorch dataloaders"""

        batch = {}
        batch["hp_imgs"] = torch.stack(
            [torch.from_numpy(sample["hp_img"]) for sample in data]
        )
        batch["hp_masks"] = torch.stack(
            [torch.from_numpy(sample["hp_mask"]) for sample in data]
        )
        batch["imgs"] = torch.stack([sample["img"] for sample in data])
        batch["masks"] = torch.stack([sample["mask"] for sample in data])
        batch["cal_infos"] = [sample["cal_info"] for sample in data]
        batch["names"] = [sample["name"] for sample in data]
        return batch


class WoodscapeHPSegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size,
        val_batch_size,
        pred_batch_size,
        train_share,
        pred_samples,
        pred_part,
        shuffle,
        shuffle_train_val_split,
        train_worker,
        val_worker,
        cam_pos,
        crop_green,
        nside,
        base_pix,
        s2_bkgd_class,
        rotate_pole,
        manual_overfit_batches,
        seed,
        version,
        training_data_fraction,
        data_fraction_seed,
    ):
        assert s2_bkgd_class in range(11)
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
        self.s2_bkgd_class = s2_bkgd_class
        self.nside = nside
        self.base_pix = base_pix
        self.pred_part = pred_part
        self.overfit_batches = manual_overfit_batches
        self.woodscape_version = version

        dataset_kwargs = {
            "nside": nside,
            "base_pix": base_pix,
            "crop_green": crop_green,
            "cam_pos": cam_pos,
            "train_share": train_share,
            "s2_bkgd_class": self.s2_bkgd_class,
            "rotate_pole": rotate_pole,
            "shuffle_train_val_split": shuffle_train_val_split,
            "woodscape_version": self.woodscape_version,
            "training_data_fraction": training_data_fraction,
            "data_fraction_seed": data_fraction_seed,
        }
        self.train_dataset = WoodscapeHPSemanticImagesDataset(
            part="train", **dataset_kwargs
        )
        self.val_dataset = WoodscapeHPSemanticImagesDataset(
            part="val", **dataset_kwargs
        )
        del dataset_kwargs["train_share"]
        pred_samples = -1 if self.overfit_batches > 0 else pred_samples
        self.pred_dataset = WoodscapeHPSemanticImagesPredictDataset(
            samples=pred_samples,
            base_train_share=train_share,
            part=pred_part,
            **dataset_kwargs,
        )

        if self.overfit_batches > 0:
            samples = self.overfit_batches * self.batch_size
            # we need to seed the RNG since the datamodule gets instantiated several times for
            # training and prediction but we need the same training indices in both cases
            # this also means that the same seed will always select the same images
            rand_gen = torch.Generator()
            rand_gen.manual_seed(seed)
            self.train_idcs = torch.randperm(
                len(self.train_dataset), generator=rand_gen
            )[:samples]

    def get_train_overfit_names(self):
        train_names = [self.train_dataset.file_names[idx] for idx in self.train_idcs]
        train_names.sort()
        return train_names

    def get_pred_overfit_sampler(self):
        train_names = self.get_train_overfit_names()
        all_pred_names = self.pred_dataset.hp_imgs_masks_dataset.file_names
        pred_idcs = [np.where(all_pred_names == n)[0].item() for n in train_names]
        assert len(pred_idcs) == len(self.train_idcs)
        return SubsetRandomSampler(pred_idcs)

    def train_dataloader(self):
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

    def val_dataloader(self):
        if self.overfit_batches > 0:
            return self.train_dataloader()
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_worker,
        )
        return val_loader

    def predict_dataloader(self):
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

    def get_img_features(self):
        """Number of channels in pictures"""
        return 3

    def get_img_dims(self):
        img, mask = self.train_dataset[0]
        return img.shape[-1]

    def get_classes(self):
        """Number of classes in segmentation (including background)"""

        num_classes = len(self.get_class_names())

        if self.s2_bkgd_class not in range(num_classes):
            return num_classes + 1
        else:
            return num_classes

    def get_class_names(self):
        seg_info = utils.get_seg_info(self.woodscape_version)
        return seg_info["class_names"]

    def get_original_img_dims(self):
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
        proj_res,
        prefix,
        top_k,
        ranking_metric,
        sort_dir,
    ):
        pred_writers = {
            None: hp_pred_writers.WoodscapeHPBasePredictionWriter,
            "base_writer": hp_pred_writers.WoodscapeHPBasePredictionWriter,
            "val_on_back_projected": hp_pred_writers.WoodscapeHPValOnBackProjectedPredictionWriter,
            "best_worst_preds": hp_pred_writers.WoodscapeHPBestWorstPredictionWriter,
            "back_projected_hp_masked_iou": WoodscapeHPBackProjectedHPMaskedIoUPredictionWriter,
        }
        kwargs = {
            "output_dir": output_dir,
            "write_interval": write_interval,
            "output_resolution": output_resolution,
            "proj_res": proj_res,
            "rotate_pole": self.rotate_pole,
            "f_out": self.get_classes(),
            "prefix": prefix,
            "nside": self.nside,
            "base_pix": self.base_pix,
            "part": self.pred_part,
            "s2_bkgd_class": self.s2_bkgd_class,
            "top_k": top_k,
            "ranking_metric": ranking_metric,
            "sort_dir": sort_dir,
            "pred_dataset": self.pred_dataset,
            "img_dims": self.get_original_img_dims(),
            "woodscape_version": self.woodscape_version,
        }
        assert (
            pred_writer_name in pred_writers.keys()
        ), f"prediction writer {pred_writer_name} unknow, implemented are {pred_writers.keys()}"

        return pred_writers[pred_writer_name](**kwargs)
