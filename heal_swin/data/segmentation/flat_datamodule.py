import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pytorch_lightning.core.datamodule import LightningDataModule

from heal_swin.data.segmentation import flat_datasets
from heal_swin.evaluation import flat_pred_writers
from heal_swin.data.segmentation import hp_datasets
from heal_swin.utils import utils


class WoodscapeSemanticImagesPredictDataset(Dataset):
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
        woodscape_version=None,
    ):
        super().__init__()

        dataset_kwargs = {
            "crop_green": crop_green,
            "cam_pos": cam_pos,
            "size": size,
            "padding": padding,
            "shuffle_train_val_split": shuffle_train_val_split,
            "woodscape_version": woodscape_version,
        }

        if isinstance(samples, float):  # samples is fraction of dataset to take
            assert 0.0 < samples <= 1.0
            if part == "train":
                train_share = base_train_share * samples
            elif part == "val":
                train_share = base_train_share + ((1 - samples) * (1 - base_train_share))
        elif isinstance(samples, int):
            dummy_ds = flat_datasets.WoodscapeSemanticImagesDataset(
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
            if samples == -1:  # special case for overfitting on a few samples: select everything
                train_share = 1.0
                part = "train"

        dataset_kwargs["train_share"] = train_share
        dataset_kwargs["part"] = part
        self.transf_sem_img_dataset = flat_datasets.WoodscapeSemanticImagesDataset(**dataset_kwargs)
        dataset_kwargs["size"] = None  # unresized images
        self.sem_img_dataset = flat_datasets.WoodscapeSemanticImagesCalibrationDataset(
            **dataset_kwargs
        )

        # take all the data for flat images and calibration, the matching is done in __getitem__
        self.hp_dataset = hp_datasets.WoodscapeHPSemanticImagesDataset(
            crop_green=crop_green,
            cam_pos=cam_pos,
            train_share=0,
            part="val",
            nside=nside,
            base_pix=base_pix,
            s2_bkgd_class=s2_bkgd_class,
            rotate_pole=rotate_pole,
            woodscape_version=woodscape_version,
        )

    def __len__(self):
        return len(self.sem_img_dataset)

    def __getitem__(self, idx):
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

    def collate_fn(self, data):
        """collate_fn to aggregate samples into batches, to be used with PyTorch dataloaders"""

        batch = {}
        batch["hp_imgs"] = torch.stack([torch.from_numpy(sample["hp_img"]) for sample in data])
        batch["hp_masks"] = torch.stack([torch.from_numpy(sample["hp_mask"]) for sample in data])
        batch["s2_imgs"] = torch.stack([sample["s2_img"] for sample in data])
        batch["s2_masks"] = torch.stack([sample["s2_mask"] for sample in data])
        batch["imgs"] = torch.stack([sample["img"] for sample in data])
        batch["masks"] = torch.stack([sample["mask"] for sample in data])
        batch["cal_infos"] = [sample["cal_info"] for sample in data]
        batch["names"] = [sample["name"] for sample in data]
        return batch


class WoodscapeFlatSegmentationDataModule(LightningDataModule):
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
        size,
        s2_bkgd_class,
        rotate_pole,
        nside,
        base_pix,
        padding,
        manual_overfit_batches,
        seed,
        version,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.pred_batch_size = pred_batch_size
        self.shuffle = shuffle
        self.train_worker = train_worker
        self.val_worker = val_worker
        self.size = size
        self.s2_bkgd_class = s2_bkgd_class
        self.base_pix = base_pix
        self.nside = nside
        self.rotate_pole = rotate_pole
        self.padding = padding
        self.overfit_batches = manual_overfit_batches
        self.pred_part = pred_part
        self.woodscape_version = version
        dataset_kwargs = {
            "crop_green": crop_green,
            "cam_pos": cam_pos,
            "train_share": train_share,
            "size": size,
            "shuffle_train_val_split": shuffle_train_val_split,
            "woodscape_version": self.woodscape_version,
        }

        dataset_kwargs["padding"] = padding
        train_dataset = val_dataset = flat_datasets.WoodscapeSemanticImagesDataset
        pred_dataset = WoodscapeSemanticImagesPredictDataset

        self.train_dataset = train_dataset(part="train", **dataset_kwargs)
        self.val_dataset = val_dataset(part="val", **dataset_kwargs)

        del dataset_kwargs["train_share"]
        pred_samples = -1 if self.overfit_batches > 0 else pred_samples
        self.pred_dataset = pred_dataset(
            samples=pred_samples,
            part=pred_part,
            base_train_share=train_share,
            nside=nside,
            base_pix=base_pix,
            **dataset_kwargs,
        )

        if self.overfit_batches > 0:
            samples = self.overfit_batches * self.batch_size
            # we need to seed the RNG since the datamodule gets instantiated several times for
            # training and prediction but we need the same training indices in both cases
            # this also means that the same seed will always select the same images
            rand_gen = torch.Generator()
            rand_gen.manual_seed(seed)
            self.train_idcs = torch.randperm(len(self.train_dataset), generator=rand_gen)[:samples]

    def get_train_overfit_names(self):
        train_names = [self.train_dataset.imgs_dataset.file_names[idx] for idx in self.train_idcs]
        train_names.sort()
        return train_names

    def get_pred_overfit_sampler(self):
        train_names = self.get_train_overfit_names()
        all_pred_names = self.pred_dataset.sem_img_dataset.file_names
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
            persistent_workers=self.train_worker > 0,
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
            persistent_workers=self.val_worker > 0,
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
            persistent_workers=self.val_worker > 0,
            collate_fn=self.pred_dataset.collate_fn,
            sampler=sampler,
        )
        return pred_loader

    def get_img_features(self):
        """Number of channels in pictures"""
        return 3

    def get_img_dims(self):
        img, mask = self.train_dataset[0]
        return tuple(img.shape[-2:])

    def get_classes(self):
        """Number of classes in segmentation (including background)"""

        num_classes = len(self.get_class_names())

        return num_classes

    def get_class_names(self):
        seg_info = utils.get_seg_info(self.woodscape_version)
        return seg_info["class_names"]

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
        proj_res,
    ):
        pred_writers = {
            None: flat_pred_writers.WoodscapeFlatBasePredictionWriter,
            "base_writer": flat_pred_writers.WoodscapeFlatBasePredictionWriter,
            "best_worst_preds": flat_pred_writers.WoodscapeFlatBestWorstPredictionWriter,
            "val_on_hp_projected": flat_pred_writers.WoodscapeFlatValOnHPProjectedPredictionWriter,
            "hp_masked_iou": flat_pred_writers.WoodscapeFlatHPMaskedIoUPredictionWriter,
        }
        cal_info = self.pred_dataset[0]["cal_info"]["intrinsic"]
        orig_size = (int(cal_info["height"]), int(cal_info["width"]))
        kwargs = {
            "output_dir": output_dir,
            "write_interval": write_interval,
            "output_resolution": output_resolution,
            "proj_res": proj_res,
            "prefix": prefix,
            "f_out": self.get_classes(),
            "part": self.pred_part,
            "top_k": top_k,
            "ranking_metric": ranking_metric,
            "sort_dir": sort_dir,
            "pred_dataset": self.pred_dataset,
            "img_dims": self.get_img_dims(),
            "nside": self.nside,
            "base_pix": self.base_pix,
            "s2_bkgd_class": self.s2_bkgd_class,
            "rotate_pole": self.rotate_pole,
            "padding": self.padding,
            "orig_size": orig_size,
            "woodscape_version": self.woodscape_version,
        }
        assert (
            pred_writer_name in pred_writers.keys()
        ), f"prediction writer {pred_writer_name} unknow, implemented are {pred_writers.keys()}"

        return pred_writers[pred_writer_name](**kwargs)
