import os
from functools import partial
import pickle
from string import Template
import re

import torch
import numpy as np
from pytorch_lightning.callbacks import BasePredictionWriter
import torchmetrics
from torchmetrics.functional import accuracy, iou
import torchvision as tv
import healpy as hp

from heal_swin.utils import utils, get_paths
from heal_swin.evaluation import custom_metrics
from heal_swin.data.segmentation.project_on_s2 import project_s2_points_to_img, sample_mask


class WoodscapeFlatBasePredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir,
        write_interval,
        output_resolution,
        prefix="",
        f_out=10,
        part="val",
        woodscape_version=None,
        **kwargs,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.resolution = output_resolution
        self.part = part
        self.acc = torchmetrics.Accuracy()
        self.iou = torchmetrics.IoU(f_out)
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
        transf_imgs = batch["s2_imgs"].cpu()
        transf_masks = batch["s2_masks"].cpu()
        preds = prediction.cpu()
        imgs = batch["imgs"].cpu()
        masks = batch["masks"].cpu()
        names = batch["names"]

        zipped_batch = zip(transf_imgs, transf_masks, preds, imgs, masks, names)
        for transf_img, transf_mask, pred, img, mask, name in zipped_batch:

            pred_overlay = utils.get_overlay(
                self.woodscape_version, pred, transf_img, mask_opacity=1
            )
            gt_overlay = utils.get_overlay(self.woodscape_version, mask, img, mask_opacity=1)

            output_size = gt_overlay.shape[-2:]
            output_size = tuple([int(dim * self.resolution) for dim in output_size])
            transform = tv.transforms.Resize(output_size)
            gt_overlay = transform(gt_overlay)
            pred_overlay = transform(pred_overlay)
            img = transform(img)

            diff = transform(pred.unsqueeze(0)) == transform(mask.unsqueeze(0))
            diff = 255 * diff.long()
            diff = torch.stack((diff, diff, diff), 0)
            diff = torch.squeeze(diff)

            overlays = torch.stack((img, diff, gt_overlay, pred_overlay)).long() / 255
            iou = self.iou(pred, transf_mask)
            path = os.path.join(
                self.output_dir, f"{self.prefix}{self.part}_{name}_best_ckpt_iou={iou:.4f}.png"
            )
            tv.utils.save_image(overlays, fp=path, nrow=2)


class WoodscapeFlatBestWorstPredictionWriter(BasePredictionWriter):
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
            "iou": partial(iou, num_classes=f_out, absent_score=0.0),
            "acc_ignored": partial(accuracy, num_classes=f_out, ignore_index=0),
            "iou_ignored": partial(iou, num_classes=f_out, absent_score=0.0, ignore_index=0),
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
        preds = prediction
        preds = preds.cpu()
        masks = batch["s2_masks"].cpu()
        names = batch["names"]

        for pred, mask, name in zip(preds, masks, names):
            metric_value = self.ranking_metric(pred, mask)
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
        all_file_names = self.pred_dataset.sem_img_dataset.file_names

        for top_bottom, names in pred_names.items():
            print(f"writing predictions of {top_bottom} {self.top_k} samples...")
            for idx, name in enumerate(names):
                sample_idx = np.where(all_file_names == name + ".png")[0].item()
                sample = self.pred_dataset[sample_idx]
                batch = sample["s2_img"].to(pl_module.device)[None, ...]
                _, pred = torch.max(pl_module(batch), 1)
                pred = pred.squeeze().cpu()
                metric_value = self.ranking_metric(pred, sample["s2_mask"])

                pred_overlay = utils.get_overlay(
                    self.woodscape_version, pred, sample["s2_img"], mask_opacity=1
                )
                gt_overlay = utils.get_overlay(
                    self.woodscape_version, sample["mask"], sample["img"], mask_opacity=1
                )

                output_size = gt_overlay.shape[-2:]
                output_size = tuple([int(dim * self.resolution) for dim in output_size])
                transform = tv.transforms.Resize(output_size)
                gt_overlay = transform(gt_overlay)
                pred_overlay = transform(pred_overlay)

                overlays = torch.stack((gt_overlay, pred_overlay)).long() / 255
                file_name = f"{self.prefix}{self.part}_{top_bottom}_{idx+1}_{name}"
                file_name += f"_{self.metric_name}={metric_value:.4f}.png"
                path = os.path.join(self.output_dir, file_name)
                tv.utils.save_image(overlays, fp=path, nrow=2)


class BasePredictionWriterIgnoreArgs(BasePredictionWriter):
    """Wraps BasePredictionWriter and ignores additional arguments in __init__"""

    def __init__(self, write_interval, **_ignore):
        super().__init__(write_interval)


class WoodscapeFlatPerCalPredictionWriter(BasePredictionWriterIgnoreArgs):
    def __init__(self, write_interval, woodscape_version, **_ignore):
        super().__init__(write_interval)
        self.woodscape_version = woodscape_version
        self.sample_lists = self.get_sample_lists()
        self.cam_pos_re = re.compile(r"^[0-9]{5,5}_(FV|RV|MVL|MVR)$")

    def get_metric_collection(self, sample_lists, metric_cls, metric_kwargs, pass_cal_info=False):
        metrics_dict = {}
        for cam_pos, cam_pos_sample_lists in sample_lists.items():
            for i, sample_list in enumerate(cam_pos_sample_lists):
                if pass_cal_info:
                    metric_kwargs["cal_info"] = sample_list["cal_info"]
                metrics_dict[f"{cam_pos}_{i}"] = metric_cls(**metric_kwargs)
        metric_collection = torchmetrics.MetricCollection(metrics_dict)
        return metric_collection

    def get_sample_lists(self):
        path = get_paths.get_datasets_path(self.woodscape_version)
        path = Template(os.path.join(path, "${cam_pos}_samples_by_cal_data.pickle"))

        sample_lists = {}
        for cam_pos in ["fv", "rv", "mvl", "mvr"]:
            with open(path.substitute(cam_pos=cam_pos), "rb") as f:
                samples_by_cal_data = pickle.load(f)
            sample_lists[cam_pos] = samples_by_cal_data

        return sample_lists

    def get_metric_key(self, file_name):
        cam_pos = self.cam_pos_re.match(file_name).group(1).lower()
        sample_lists = self.sample_lists[cam_pos]
        cal_idx = None
        for idx in range(len(sample_lists)):
            if np.sum(sample_lists[idx]["file_names"] == file_name) > 0:
                cal_idx = idx
                break
        assert cal_idx is not None, f"{file_name} not found in calibration data"
        return f"{cam_pos}_{cal_idx}"


class WoodscapeFlatHPMaskedIoUPredictionWriter(WoodscapeFlatPerCalPredictionWriter):

    METRIC_NAME = "hp_masked_iou"

    def __init__(
        self,
        output_dir,
        write_interval,
        img_dims,
        f_out=10,
        rotate_pole=False,
        base_pix=8,
        nside=256,
        part="val",
        prefix="",
        woodscape_version=None,
        **_ignore,
    ):
        super().__init__(write_interval, woodscape_version)
        self.f_out = f_out
        self.prefix = prefix
        self.part = part
        metric_kwargs = {
            "base_pix": base_pix,
            "nside": nside,
            "proj_res": img_dims,
            "rotate_pole": rotate_pole,
            "num_classes": f_out,
        }
        self.metrics = self.get_metric_collection(
            self.sample_lists, custom_metrics.HPMaskedIoU, metric_kwargs, pass_cal_info=True
        )

    def on_predict_epoch_start(self, trainer, pl_module):
        for metric in self.metrics.values():
            metric.to(pl_module.device)

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
        preds = prediction.to(pl_module.device)
        masks = batch["s2_masks"].to(pl_module.device)
        names = batch["names"]

        for pred, mask, name in zip(preds, masks, names):
            metric_key = self.get_metric_key(name)
            self.metrics[metric_key].update(pred[None, ...], mask[None, ...])

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        mean_iou = torchmetrics.IoU(num_classes=self.f_out, reduction="none")
        for key, metric in self.metrics.items():
            if metric.iou.confmat.sum().item() > 0:
                confmat = metric.iou.confmat.cpu()
                mean_iou.confmat += confmat

        class_ious = mean_iou.compute()
        iou = torch.mean(class_ious).item()
        name = f"{self.prefix}{self.part}_{self.METRIC_NAME}"
        trainer.logger.log_metrics({name: iou})


class WoodscapeFlatValOnHPProjectedPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir,
        write_interval,
        part="",
        prefix="",
        rotate_pole=False,
        f_out=10,
        nside=256,
        base_pix=8,
        s2_bkgd_class=0,
        orig_size=None,
        padding=[0, 0, 0, 0],
        **_ignore,
    ):
        super().__init__(write_interval=write_interval)
        self.rotate_pole = rotate_pole
        self.f_out = f_out
        self.nside = nside
        self.base_pix = base_pix
        self.s2_bkgd_class = s2_bkgd_class
        self.part = part

        self.acc = torchmetrics.Accuracy()
        self.acc_ignored = torchmetrics.Accuracy(ignore_index=0)
        self.iou = torchmetrics.IoU(num_classes=f_out, reduction="none")

        self.mask_transform = (
            utils.id if orig_size is None else tv.transforms.Resize(orig_size, interpolation=0)
        )
        neg_padding = [-p for p in padding]
        self.padding = tv.transforms.Pad(neg_padding, fill=0)

        if prefix != "" and prefix[-1] != "_":
            prefix = prefix + "_"

        self.prefix = prefix

        self.set_theta_phi(nside, base_pix)

    def set_theta_phi(self, nside, base_pix):
        npix = hp.pixelfunc.nside2npix(nside)
        ipix = np.arange(npix)
        theta, phi = hp.pixelfunc.pix2ang(nside, ipix, nest=True)

        half_idcs = np.arange(npix * base_pix // 12)
        self.theta = theta[half_idcs]
        self.phi = phi[half_idcs]

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
        flat_preds = prediction

        # undo transforms since ground truth is projected from originals
        flat_preds = self.padding(flat_preds)
        flat_preds = self.mask_transform(flat_preds)
        flat_preds = flat_preds.cpu()

        hp_masks = batch["hp_masks"]
        cal_infos = batch["cal_infos"]

        hp_preds = []
        for idx, (flat_pred, cal_info) in enumerate(zip(flat_preds, cal_infos)):
            u, v = project_s2_points_to_img(self.theta, self.phi, cal_info, self.rotate_pole)
            pred = sample_mask(flat_pred.numpy(), v, u, s2_bkgd_class=self.s2_bkgd_class)
            # add trivial batch dimension:
            pred = torch.from_numpy(pred)[None, ...]
            hp_preds.append(pred)
            mask = hp_masks[idx][None, ...]
            self.acc.update(pred, mask)
            self.acc_ignored.update(pred, mask)
            self.iou.update(pred, mask)

        hp_pred_batch = torch.cat(hp_preds, dim=0)
        return hp_pred_batch, hp_masks

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        pref = f"{self.prefix}{self.part}"
        suff = "projected_to_hp"
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
