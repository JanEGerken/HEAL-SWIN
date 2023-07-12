import os
from functools import partial

import numpy as np
import torch
import torchvision as tv
from torchvision.transforms.functional import InterpolationMode
from pytorch_lightning.callbacks import BasePredictionWriter
import torchmetrics
from torchmetrics.functional import accuracy, iou

from heal_swin.data.segmentation.project_on_s2 import project_hp_img_back, project_hp_mask_back
from heal_swin.utils import utils
from heal_swin.evaluation.flat_pred_writers import (
    BasePredictionWriterIgnoreArgs,
    WoodscapeFlatHPMaskedIoUPredictionWriter,
)


class WoodscapeHPBasePredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir,
        write_interval,
        output_resolution=1,
        rotate_pole=False,
        f_out=10,
        prefix="",
        nside=256,
        base_pix=8,
        s2_bkgd_class=0,
        part="val",
        woodscape_version=None,
        **kwargs,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.resolution = output_resolution
        self.rotate_pole = rotate_pole
        self.nside = nside
        self.base_pix = base_pix
        self.s2_bkgd_class = s2_bkgd_class
        self.part = part
        self.iou = partial(iou, num_classes=f_out, absent_score=float("nan"), reduction="none")
        self.woodscape_version = woodscape_version

        if prefix != "" and prefix[-1] != "_":
            prefix = prefix + "_"

        self.prefix = prefix

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        hp_preds = prediction
        hp_imgs = batch["hp_imgs"].cpu()
        hp_preds = hp_preds.cpu()
        hp_masks = batch["hp_masks"].cpu()
        imgs = batch["imgs"].cpu()
        masks = batch["masks"].cpu()
        cal_infos = batch["cal_infos"]
        names = batch["names"]

        zipped_batch = zip(hp_imgs, hp_preds, hp_masks, imgs, masks, cal_infos, names)
        for hp_img, hp_pred, hp_mask, img, mask, cal_info, name in zipped_batch:
            img_new = project_hp_img_back(
                hp_img.numpy(),
                cal_info,
                self.resolution,
                self.rotate_pole,
                self.base_pix,
            )
            pred = project_hp_mask_back(
                hp_pred.numpy(),
                cal_info,
                self.resolution,
                self.rotate_pole,
                self.nside,
                self.base_pix,
                self.s2_bkgd_class,
            )
            img_new = torch.from_numpy(img_new)
            pred = torch.from_numpy(pred)

            pred_overlay = utils.get_overlay(
                self.woodscape_version, pred, img_new, mask_opacity=0.7
            )
            gt_overlay = utils.get_overlay(self.woodscape_version, mask, img, mask_opacity=0.7)

            output_size = (pred_overlay.shape[-2], pred_overlay.shape[-1])
            gt_overlay_transform = tv.transforms.Resize(output_size)
            gt_overlay = gt_overlay_transform(gt_overlay)

            overlays = torch.stack((gt_overlay, pred_overlay)).long() / 255
            class_ious = self.iou(hp_pred, hp_mask)
            iou = utils.nanmean(class_ious)
            path = os.path.join(
                self.output_dir, f"{self.prefix}{self.part}_{name}_iou={iou:.4f}.png"
            )
            tv.utils.save_image(overlays, fp=path, nrow=2)


class WoodscapeHPValOnBackProjectedPredictionWriter(BasePredictionWriterIgnoreArgs):
    def __init__(
        self,
        output_dir,
        write_interval,
        img_dims,
        rotate_pole=False,
        f_out=10,
        prefix="",
        nside=256,
        base_pix=8,
        s2_bkgd_class=0,
        part="val",
        ignore_gt_classes=[],
        woodscape_version=None,
        proj_res=1.0,
        **_ignore,
    ):
        super().__init__(
            output_dir=output_dir,
            write_interval=write_interval,
            img_dims=img_dims,
            part=part,
            prefix=prefix,
            ignore_gt_classes=ignore_gt_classes,
            woodscape_version=woodscape_version,
        )
        self.rotate_pole = rotate_pole
        self.f_out = f_out
        self.nside = nside
        self.base_pix = base_pix
        self.s2_bkgd_class = s2_bkgd_class
        self.part = part
        self.proj_res = proj_res
        self.org_size = None

        self.acc = torchmetrics.Accuracy()
        self.acc_ignored = torchmetrics.Accuracy(ignore_index=0)
        self.iou = torchmetrics.IoU(num_classes=f_out, reduction="none")

        if prefix != "" and prefix[-1] != "_":
            prefix = prefix + "_"

        self.mask_transform = tv.transforms.Resize(
            proj_res, interpolation=InterpolationMode.NEAREST
        )

        self.prefix = prefix

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        assert (
            pl_module.global_rank == 0
        ), "currently the back-projected validation works only on a single GPU"
        hp_preds = prediction
        hp_preds = hp_preds.cpu()
        masks = self.mask_transform(batch["masks"])
        cal_infos = batch["cal_infos"]
        if self.org_size is None:
            cal_info = cal_infos[0]["intrinsic"]
            self.org_size = int(cal_info["height"]), int(cal_info["width"])

        flat_preds = []
        for idx, (hp_pred, cal_info) in enumerate(zip(hp_preds, cal_infos)):
            pred = project_hp_mask_back(
                hp_pred.numpy(),
                cal_info,
                output_resolution=self.proj_res,
                rotate_pole=self.rotate_pole,
                nside=self.nside,
                base_pix=self.base_pix,
                s2_bkgd_class=self.s2_bkgd_class,
            )
            # add trivial batch dimension:
            pred = torch.from_numpy(pred)[None, ...]
            flat_preds.append(pred)
            mask = masks[idx][None, ...]
            self.acc.update(pred, mask)
            self.acc_ignored.update(pred, mask)
            self.iou.update(pred, mask)

        flat_pred_batch = torch.cat(flat_preds, dim=0)
        return flat_pred_batch, masks

    def get_res_suff(self):
        if isinstance(self.proj_res, int) and self.proj_res != min(self.org_size):
            return f"_res_{self.proj_res}"
        elif isinstance(self.proj_res, tuple):
            return f"_res_{self.proj_res[0]}_{self.proj_res[1]}"
        else:
            return ""

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        pref = f"{self.prefix}{self.part}"
        suff = "back_projected" + self.get_res_suff()

        class_ious = self.iou.compute()
        iou = torch.mean(class_ious).item()
        iou_ignored = torch.mean(class_ious[1:]).item()
        metrics = {
            f"{pref}_acc_{suff}": self.acc.compute().item(),
            f"{pref}_acc_ignored_{suff}": self.acc_ignored.compute().item(),
            f"{pref}_iou_{suff}": iou,
            f"{pref}_iou_ignored_{suff}": iou_ignored,
        }
        trainer.logger.log_metrics(metrics)


class WoodscapeHPBestWorstPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir,
        write_interval,
        output_resolution=1,
        rotate_pole=False,
        f_out=10,
        prefix="",
        nside=256,
        base_pix=8,
        s2_bkgd_class=0,
        part="val",
        top_k=5,
        ranking_metric="iou_ignored",
        sort_dir="asc",
        pred_dataset=None,
        woodscape_version=None,
        **_ignore,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.resolution = output_resolution
        self.rotate_pole = rotate_pole
        self.nside = nside
        self.base_pix = base_pix
        self.s2_bkgd_class = s2_bkgd_class
        self.part = part
        self.top_k = top_k
        self.pred_dataset = pred_dataset
        self.woodscape_version = woodscape_version
        metrics = {
            "acc": partial(accuracy, num_classes=f_out),
            "iou": partial(self.iou, num_classes=f_out),
            "acc_ignored": partial(accuracy, num_classes=f_out, ignore_index=0),
            "iou_ignored": partial(self.iou, num_classes=f_out, ignore=True),
        }
        assert ranking_metric in list(metrics.keys())
        self.ranking_metric = metrics[ranking_metric]
        self.metric_name = ranking_metric
        assert sort_dir in ["asc", "desc"]
        self.sort_dir = sort_dir

        self.metric_values = []
        self.names = []

        if prefix != "" and prefix[-1] != "_":
            prefix = prefix + "_"

        self.prefix = prefix

    def iou(self, preds, targets, num_classes, ignore=False):
        class_ious = iou(
            preds, targets, num_classes=num_classes, absent_score=float("nan"), reduction="none"
        )
        if ignore:
            class_ious = class_ious[1:]

        return utils.nanmean(class_ious)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        hp_preds = prediction
        hp_preds = hp_preds.cpu()
        hp_masks = batch["hp_masks"].cpu()
        names = batch["names"]

        for hp_pred, hp_mask, name in zip(hp_preds, hp_masks, names):
            metric_value = self.ranking_metric(hp_pred, hp_mask)
            self.metric_values.append(metric_value.item())
            self.names.append(name)

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        self.metric_values = np.array(self.metric_values)
        self.names = np.array(self.names)

        sort_idcs = np.argsort(self.metric_values)
        if self.sort_dir == "desc":
            sort_idcs = sort_idcs[::-1]

        pred_names = {
            "top": self.names[sort_idcs[-self.top_k :]][::-1],
            "bottom": self.names[sort_idcs[: self.top_k]],
        }
        all_file_names = self.pred_dataset.hp_imgs_masks_dataset.file_names

        for top_bottom, names in pred_names.items():
            print(f"writing predictions of {top_bottom} {self.top_k} samples...")
            for idx, name in enumerate(names):
                sample_idx = np.where(all_file_names == name + ".npz")[0].item()
                sample = self.pred_dataset[sample_idx]
                batch = torch.from_numpy(sample["hp_img"]).to(pl_module.device)[None, ...]
                _, hp_pred = torch.max(pl_module(batch), 1)
                hp_pred = hp_pred.squeeze().cpu()
                metric_value = self.ranking_metric(hp_pred, torch.from_numpy(sample["hp_mask"]))

                img_new = project_hp_img_back(
                    sample["hp_img"],
                    sample["cal_info"],
                    self.resolution,
                    self.rotate_pole,
                    self.base_pix,
                )
                pred = project_hp_mask_back(
                    hp_pred.numpy(),
                    sample["cal_info"],
                    self.resolution,
                    self.rotate_pole,
                    self.nside,
                    self.base_pix,
                    self.s2_bkgd_class,
                )
                img_new = torch.from_numpy(img_new)
                pred = torch.from_numpy(pred)

                pred_overlay = utils.get_overlay(
                    self.woodscape_version, pred, img_new, mask_opacity=0.7
                )
                gt_overlay = utils.get_overlay(
                    self.woodscape_version, sample["mask"], sample["img"], mask_opacity=0.7
                )

                output_size = (pred_overlay.shape[-2], pred_overlay.shape[-1])
                gt_overlay_transform = tv.transforms.Resize(output_size)
                gt_overlay = gt_overlay_transform(gt_overlay)

                overlays = torch.stack((gt_overlay, pred_overlay)).long() / 255
                file_name = f"{self.prefix}{self.part}_{top_bottom}_{idx+1}_{name}"
                file_name += f"_{self.metric_name}={metric_value:.4f}.png"
                path = os.path.join(self.output_dir, file_name)
                tv.utils.save_image(overlays, fp=path, nrow=2)


class WoodscapeHPBackProjectedHPMaskedIoUPredictionWriter(WoodscapeFlatHPMaskedIoUPredictionWriter):

    METRIC_NAME = "back_projected_hp_masked_iou"

    def __init__(
        self,
        output_dir,
        write_interval,
        img_dims,
        part="",
        prefix="",
        rotate_pole=False,
        f_out=10,
        nside=256,
        base_pix=8,
        s2_bkgd_class=0,
        orig_size=None,
        padding=[0, 0, 0, 0],
        woodscape_version=None,
        proj_res=1.0,
        **_ignore,
    ):
        super().__init__(
            output_dir=output_dir,
            write_interval=write_interval,
            img_dims=proj_res,
            f_out=f_out,
            rotate_pole=rotate_pole,
            base_pix=base_pix,
            nside=nside,
            part=part,
            prefix=prefix,
            woodscape_version=woodscape_version,
        )
        self.proj_pred_writer = WoodscapeHPValOnBackProjectedPredictionWriter(
            output_dir=output_dir,
            write_interval=write_interval,
            img_dims=img_dims,
            rotate_pole=rotate_pole,
            f_out=f_out,
            prefix=prefix,
            nside=nside,
            base_pix=base_pix,
            s2_bkgd_class=s2_bkgd_class,
            part=part,
            woodscape_version=woodscape_version,
            proj_res=proj_res,
        )

        self.target_mask_transform = tv.transforms.Resize(
            proj_res, interpolation=InterpolationMode.NEAREST
        )

        self.METRIC_NAME += self.proj_pred_writer.get_res_suff()

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        flat_preds, flat_masks = self.proj_pred_writer.write_on_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            prediction=prediction,
            batch_indices=batch_indices,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

        flat_masks = self.target_mask_transform(flat_masks)

        flat_batch = {"s2_masks": flat_masks, "names": batch["names"]}

        super().write_on_batch_end(
            trainer, pl_module, flat_preds, batch_indices, flat_batch, batch_idx, dataloader_idx
        )
