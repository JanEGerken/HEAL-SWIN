from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from pytorch_lightning.core.datamodule import LightningDataModule

from heal_swin.data.depth_estimation import flat_depth_datasets
from heal_swin.data.depth_estimation import normalize_depth_data
from heal_swin.evaluation.flat_depth_pred_writers import WoodscapeDepthFlatBasePredictionWriter
from heal_swin.evaluation import flat_depth_pred_writers


class WoodscapeFlatDepthDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size,
        val_batch_size,
        train_share,
        pred_part,
        train_worker,
        val_worker,
        bandwidth,
        manual_overfit_batches,
        shuffle,
        shuffle_train_val_split,
        cam_pos,
        nside,
        base_pix,
        size,
        padding,
        pred_samples,
        seed,
        data_transform,
        mask_background,
        rotate_pole,
        pred_batch_size,
        normalize_data,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.overfit_batches = manual_overfit_batches
        self.train_worker = train_worker
        self.shuffle = shuffle
        self.pred_part = pred_part
        self.nside = nside
        self.base_pix = base_pix
        self.rotate_pole = rotate_pole
        self.padding = padding
        self.pred_batch_size = pred_batch_size
        self.bandwidth = bandwidth

        self.val_batch_size = val_batch_size
        self.val_worker = val_worker

        self.data_transform = data_transform
        self.mask_background = mask_background

        self.normalize_data = normalize_data
        self.depth_data_statistics = (
            normalize_depth_data.MaskedDepthDataStatistics()
            if mask_background
            else normalize_depth_data.DepthDataStatistics()
        )

        dataset_kwargs = {
            "cam_pos": cam_pos,
            "train_share": train_share,
            "size": size,
            "shuffle_train_val_split": shuffle_train_val_split,
            "padding": padding,
        }

        train_dataset = val_dataset = flat_depth_datasets.WoodscapeDepthDataset
        pred_dataset = flat_depth_datasets.WoodscapeDepthImagesPredictDataset

        self.train_dataset = train_dataset(
            part="train",
            data_transform=data_transform,
            mask_background=mask_background,
            normalize_data=normalize_data,
            **dataset_kwargs,
        )
        self.val_dataset = val_dataset(
            part="val",
            data_transform=data_transform,
            mask_background=mask_background,
            normalize_data=normalize_data,
            **dataset_kwargs,
        )

        del dataset_kwargs["train_share"]
        pred_samples = -1 if self.overfit_batches > 0 else pred_samples
        self.pred_dataset = pred_dataset(
            samples=pred_samples,
            part=pred_part,
            base_train_share=train_share,
            nside=nside,
            base_pix=base_pix,
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
        train_names = [self.train_dataset.imgs_dataset.file_names[idx] for idx in self.train_idcs]
        train_names.sort()
        return train_names

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
        return tuple(img.shape[-2:])

    def get_classes(self) -> int:
        """For compatibility, not relevant for depth regression"""
        return 1

    def get_pred_overfit_sampler(self) -> SubsetRandomSampler:
        train_names = self.get_train_overfit_names()
        all_pred_names = self.pred_dataset.sem_img_dataset.file_names
        pred_idcs = [np.where(all_pred_names == n)[0].item() for n in train_names]
        assert len(pred_idcs) == len(self.train_idcs)
        return SubsetRandomSampler(pred_idcs)

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
    ) -> Union[WoodscapeDepthFlatBasePredictionWriter]:

        pred_writers = {
            None: flat_depth_pred_writers.WoodscapeDepthFlatBasePredictionWriter,
            "base_writer": flat_depth_pred_writers.WoodscapeDepthFlatBasePredictionWriter,
            "val_on_hp_projected": (
                flat_depth_pred_writers.WoodscapeDepthFlatValOnHPProjectedPredictionWriter
            ),
            "best_worst_preds": flat_depth_pred_writers.WoodscapeDepthFlatBestWorstPredictionWriter,
            "chamfer_distance": (
                flat_depth_pred_writers.WoodscapeDepthFlatChamferDistBestWorstPredictionWriter
            ),
        }
        cal_info = self.pred_dataset[0]["cal_info"]["intrinsic"]
        orig_size = (int(cal_info["height"]), int(cal_info["width"]))
        self.s2_bkgd_class = -1
        kwargs = {
            "output_dir": output_dir,
            "write_interval": write_interval,
            "output_resolution": output_resolution,
            "prefix": prefix,
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
            "data_transform": self.data_transform,
            "mask_background": self.mask_background,
            "normalize_data": self.normalize_data,
        }
        assert (
            pred_writer_name in pred_writers.keys()
        ), f"prediction writer {pred_writer_name} unknow, implemented are {pred_writers.keys()}"

        return pred_writers[pred_writer_name](**kwargs)
