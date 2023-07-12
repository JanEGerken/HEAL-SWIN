import os
import copy

import torch
import numpy as np
from pytorch_lightning.callbacks import BasePredictionWriter
import torchvision as tv
import matplotlib.pyplot as plt
import healpy as hp

from heal_swin.evaluation import custom_metrics
from heal_swin.data.depth_estimation import normalize_depth_data
from heal_swin.utils import depth_utils, utils
from heal_swin.data.segmentation.project_on_s2 import project_s2_points_to_img
from heal_swin.data.depth_estimation.project_depth_on_s2 import sample_mask


class WoodscapeDepthFlatBasePredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir,
        write_interval,
        output_resolution=1,
        prefix="",
        part="val",
        data_transform=False,
        mask_background=False,
        normalize_data=None,
        interpolation_mode="nearest",
        **_ignore,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.resolution = output_resolution
        self.part = part
        self.mse = custom_metrics.DepthMSE()

        self.data_transform = data_transform
        self.mask_background = mask_background
        self.normalize_data = normalize_data

        self.depth_data_statistics = normalize_depth_data.get_depth_data_stats(
            data_transform=self.data_transform, mask_background=self.mask_background
        )
        self.interpolation_modes = {
            "nearest": tv.transforms.InterpolationMode.NEAREST,
            "bilinear": tv.transforms.InterpolationMode.BILINEAR,
        }
        self.interpolation_mode = interpolation_mode

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

            transf_mask[transf_mask.isinf()] = float("nan")
            transf_mask = depth_utils.unnormalize_and_retransform(
                data=transf_mask,
                normalization=self.normalize_data,
                data_stats=self.depth_data_statistics,
                data_transform=self.data_transform,
            )

            if not pl_module.use_logvar:
                pred_std = None
            else:
                pred_std = torch.sqrt(torch.exp(pred[1, :]))

            pred = pred.unsqueeze(0)
            transf_mask = transf_mask.unsqueeze(0)

            mse = self.mse(pred, transf_mask)

            pred_mean = pred[:, 0, ...]
            ranged_mse_map = depth_utils.create_ranged_mse_mask(
                pl_module.metric_dict, pred, transf_mask
            )

            original_size = mask.shape[-2:]
            output_size = tuple([int(dim * 1) for dim in original_size])
            transform = tv.transforms.Resize(
                output_size, interpolation=self.interpolation_modes[self.interpolation_mode]
            )

            pred_mean = transform(pred_mean.unsqueeze(0)).squeeze()
            transf_mask = transform(transf_mask.unsqueeze(0)).squeeze()
            img = transform(img)

            if pred_std is not None:
                pred_std = transform(pred_std)

            plt_path = os.path.join(
                self.output_dir,
                f"{self.prefix}{self.part}_{name}_best_ckpt_mse={mse:.4f}.png",
            )
            depth_utils.save_depth_pred_comparison_image(
                pred_dist=pred_mean,
                original_ground_truth=mask,
                ground_truth_after_transforms=transf_mask,
                img=img,
                filepath=plt_path,
                metric_dict=pl_module.metric_dict,
                std=pred_std,
                ranged_mse_map=ranged_mse_map,
            )


class WoodscapeDepthFlatValOnHPProjectedPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir,
        write_interval,
        part="",
        prefix="",
        rotate_pole=False,
        nside=256,
        base_pix=8,
        orig_size=None,
        padding=[0, 0, 0, 0],
        output_resolution=1,
        data_transform=False,
        mask_background=False,
        normalize_data=None,
        interpolation_mode="nearest",
        **_ignore,
    ):
        super().__init__(write_interval=write_interval)
        self.rotate_pole = rotate_pole
        self.nside = nside
        self.base_pix = base_pix
        self.part = part
        self.resolution = output_resolution

        self.interpolation_modes = {
            "nearest": tv.transforms.InterpolationMode.NEAREST,
            "bilinear": tv.transforms.InterpolationMode.BILINEAR,
        }
        self.interpolation_mode = interpolation_mode

        self.mask_transform = (
            utils.id
            if orig_size is None
            else tv.transforms.Resize(
                orig_size, interpolation=self.interpolation_modes[self.interpolation_mode]
            )
        )
        neg_padding = [-p for p in padding]
        self.padding = tv.transforms.Pad(neg_padding, fill=0)

        self.data_transform = data_transform
        self.mask_background = mask_background
        self.normalize_data = normalize_data

        self.depth_data_statistics = normalize_depth_data.get_depth_data_stats(
            data_transform=self.data_transform, mask_background=self.mask_background
        )
        self.metric_dict = {
            "mse": custom_metrics.DepthMSE(),
            "SILogE": custom_metrics.ScaleInvariantLogError(),
            "iRMSE": custom_metrics.DepthiRMSE(),
            "RelAE": custom_metrics.DepthRelAE(total_mean=self.depth_data_statistics.mean),
            "RelSE": custom_metrics.DepthRelSE(total_mean=self.depth_data_statistics.mean),
        }

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

        # Get predictions, already in [m] units, no need for renormalisation
        flat_preds = prediction

        # undo transforms since ground truth is projected from originals
        flat_preds = self.padding(flat_preds)
        flat_preds = self.mask_transform(flat_preds)
        flat_preds = flat_preds.cpu()

        hp_masks = batch["hp_masks"]  # This is retrieved normalized and transformed (while in hp)
        hp_masks = depth_utils.transform_and_normalize(
            data=hp_masks,
            normalization=self.normalize_data,
            data_stats=self.depth_data_statistics,
            data_transform=self.data_transform,
        )
        cal_infos = batch["cal_infos"]

        hp_preds = []
        for idx, (flat_pred, cal_info) in enumerate(zip(flat_preds, cal_infos)):
            pred_mean = flat_pred
            u, v = project_s2_points_to_img(self.theta, self.phi, cal_info, self.rotate_pole)
            pred_mean = sample_mask(pred_mean.numpy(), v, u, s2_bkgd_class=float("nan"))
            # add trivial batch dimension:
            pred_mean = torch.from_numpy(pred_mean)[None, ...]
            hp_preds.append(pred_mean)
            mask = hp_masks[idx][None, ...]
            for met_key in self.metric_dict:
                self.metric_dict[met_key].update(pred_mean, mask)

        hp_pred_batch = torch.cat(hp_preds, dim=0)
        return hp_pred_batch, hp_masks

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        pref = f"{self.prefix}{self.part}"
        suff = "projected_to_hp"
        metrics = {}
        for met_key in self.metric_dict:
            metrics[pref + "_" + met_key + "_" + suff] = self.metric_dict[met_key].compute().item()
        trainer.logger.log_metrics(metrics)


class WoodscapeDepthFlatBestWorstPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir,
        write_interval,
        output_resolution=1,
        rotate_pole=False,
        prefix="",
        nside=256,
        base_pix=8,
        part="val",
        top_k=5,
        ranking_metric="mse",
        sort_dir="desc",
        pred_dataset=None,
        woodscape_version=None,
        data_transform=False,
        mask_background=False,
        normalize_data=None,
        interpolation_mode="nearest",
        **_ignore,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.resolution = output_resolution
        self.rotate_pole = rotate_pole
        self.nside = nside
        self.base_pix = base_pix
        self.part = part
        self.top_k = top_k
        self.pred_dataset = pred_dataset
        self.woodscape_version = woodscape_version

        self.data_transform = data_transform
        self.mask_background = mask_background
        self.normalize_data = normalize_data

        self.depth_data_statistics = normalize_depth_data.get_depth_data_stats(
            data_transform=self.data_transform, mask_background=self.mask_background
        )

        self.metrics = {
            "mse": custom_metrics.DepthMSE(),
            "SILogE": custom_metrics.ScaleInvariantLogError(),
            "iRMSE": custom_metrics.DepthiRMSE(),
            "RelAE": custom_metrics.DepthRelAE(total_mean=self.depth_data_statistics.mean),
            "RelSE": custom_metrics.DepthRelSE(total_mean=self.depth_data_statistics.mean),
        }
        assert ranking_metric in list(self.metrics.keys())
        self.ranking_metric = self.metrics[ranking_metric]
        self.metric_name = ranking_metric
        assert sort_dir in ["asc", "desc"]
        self.sort_dir = sort_dir

        self.metric_values = []
        self.names = []

        if prefix != "" and prefix[-1] != "_":
            prefix = prefix + "_"

        self.prefix = prefix

        self.interpolation_modes = {
            "nearest": tv.transforms.InterpolationMode.NEAREST,
            "bilinear": tv.transforms.InterpolationMode.BILINEAR,
        }
        self.interpolation_mode = interpolation_mode

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
        preds = prediction  # This is in correct units as per output
        preds = preds.cpu()
        masks = batch["s2_masks"].cpu()  # This should be normalized and transformed back
        names = batch["names"]

        for pred, mask, name in zip(preds, masks, names):
            mask = depth_utils.unnormalize_and_retransform(
                data=mask,
                normalization=self.normalize_data,
                data_stats=self.depth_data_statistics,
                data_transform=self.data_transform,
            )
            pred = pred.unsqueeze(0)

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
        all_file_names = self.pred_dataset.hp_dataset.file_names

        for top_bottom, names in pred_names.items():
            print(f"writing predictions of {top_bottom} {self.top_k} samples...")
            for idx, name in enumerate(names):
                sample_idx = np.where(all_file_names == name + ".npz")[0].item()
                sample = self.pred_dataset[sample_idx]
                batch = sample["s2_img"].to(pl_module.device)[None, ...]
                pred_mean = pl_module(batch)  # Get the model output, units [m]
                pred_mean = pred_mean.squeeze().cpu()

                transf_mask = sample["s2_mask"]
                transf_mask[transf_mask.isinf()] = float("nan")

                transf_mask = depth_utils.unnormalize_and_retransform(
                    data=transf_mask,
                    normalization=self.normalize_data,
                    data_stats=self.depth_data_statistics,
                    data_transform=self.data_transform,
                )

                pred_mean = pred_mean.unsqueeze(0)
                pred_mean = pred_mean.unsqueeze(0)
                metric_value = self.ranking_metric(pred_mean, transf_mask)
                pred_mean = pred_mean.squeeze()

                img = sample["s2_img"]

                output_size = (sample["mask"].shape[-2], sample["mask"].shape[-1])
                og_resize = tv.transforms.Resize(
                    output_size, interpolation=self.interpolation_modes[self.interpolation_mode]
                )
                transf_mask = og_resize(transf_mask.unsqueeze(0)).squeeze()
                pred_mean = og_resize(pred_mean.unsqueeze(0)).squeeze()

                pred_mean = pred_mean.unsqueeze(0)
                pred_mean = pred_mean.unsqueeze(0)
                metric_value_post_resize = self.ranking_metric(pred_mean, transf_mask)
                pred_mean = pred_mean.squeeze()

                file_name = (
                    f"{self.prefix}{self.part}_{self.metric_name}_{top_bottom}_{idx+1}_{name}"
                )
                file_name += (
                    f"_{self.metric_name}={metric_value:.4f}_{metric_value_post_resize:.4f}.png"
                )
                path = os.path.join(self.output_dir, file_name)
                depth_utils.save_depth_pred_comparison_image(
                    pred_dist=pred_mean,
                    original_ground_truth=sample["mask"],
                    ground_truth_after_transforms=transf_mask,
                    img=img,
                    filepath=path,
                    metric_dict=pl_module.metric_dict,
                    std=None,
                )


class WoodscapeDepthFlatChamferDistBestWorstPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir,
        write_interval,
        output_resolution=1,
        rotate_pole=False,
        prefix="",
        nside=256,
        base_pix=8,
        part="val",
        top_k=2,
        ranking_metric="chamfer_distance",
        sort_dir="desc",
        pred_dataset=None,
        woodscape_version=None,
        data_transform=False,
        mask_background=False,
        normalize_data=None,
        interpolation_mode="nearest",
        **_ignore,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.resolution = output_resolution
        self.rotate_pole = rotate_pole
        self.nside = nside
        self.base_pix = base_pix
        self.part = part
        self.top_k = top_k
        self.pred_dataset = pred_dataset

        self.woodscape_version = woodscape_version

        self.data_transform = data_transform
        self.mask_background = mask_background
        self.normalize_data = normalize_data

        self.depth_data_statistics = normalize_depth_data.get_depth_data_stats(
            data_transform=self.data_transform, mask_background=self.mask_background
        )

        self.metrics = {
            "chamfer_distance": custom_metrics.ChamferDistance(),
            "chamfer_distance_full_res": custom_metrics.ChamferDistance(),
            "chamfer_distance_full_res_hp_masked": custom_metrics.ChamferDistance(),
            "chamfer_distance_small_res_hp_masked": custom_metrics.ChamferDistance(),
        }
        self.culuminative_metrics = {
            "culum_" + key: copy.deepcopy(self.metrics[key]) for key in self.metrics.keys()
        }

        assert ranking_metric in list(self.metrics.keys())
        self.ranking_metric = self.metrics[ranking_metric]
        self.metric_name = ranking_metric
        assert sort_dir in ["asc", "desc"]
        self.sort_dir = sort_dir

        self.metric_values = []
        self.names = []

        if prefix != "" and prefix[-1] != "_":
            prefix = prefix + "_"

        self.prefix = prefix

        self.interpolation_modes = {
            "nearest": tv.transforms.InterpolationMode.NEAREST,
            "bilinear": tv.transforms.InterpolationMode.BILINEAR,
        }
        self.interpolation_mode = interpolation_mode

        self.small_res_transformation = tv.transforms.Resize(
            (629, 834), interpolation=self.interpolation_modes[self.interpolation_mode]
        )  # resize to (629,834) results in 0.05% more pixels than the hp resolution
        # and the aspect ratio is 99.93% of the original aspect ratio

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
        preds = prediction  # This is in correct units as per output
        preds = preds
        masks = batch["s2_masks"]  # This should be normalized and transformed
        full_res_masks = batch["masks"]
        names = batch["names"]
        cal_infos = batch["cal_infos"]

        for ind, (pred, mask, name, cal_info, full_res_mask) in enumerate(
            zip(preds, masks, names, cal_infos, full_res_masks)
        ):
            mask = depth_utils.unnormalize_and_retransform(
                data=mask,
                normalization=self.normalize_data,
                data_stats=self.depth_data_statistics,
                data_transform=self.data_transform,
            )

            pred = pred[
                None, ...
            ]  # pred has shape [c_out, H, W] to be converted to [1, c_out, H, W]
            mask = mask[None, ...]  # mask has shape [H,W] to be converted to [1, H, W]
            mask[mask.isinf()] = float("nan")

            full_res_mask = full_res_mask[None, ...]
            if self.mask_background:
                gt_foreground = depth_utils.get_foreground_mask(
                    mask, background_val=(float("nan"), float("inf"), 1000)
                )
                full_res_gt_foreground = depth_utils.get_foreground_mask(
                    full_res_mask, background_val=(float("nan"), float("inf"), 1000)
                )
            else:
                gt_foreground = None
                full_res_gt_foreground = None

            metric_value = self.ranking_metric(
                pred,
                mask,
                cal_info,
                hp_data=False,
                foreground_pix=gt_foreground,
                nside=self.nside,
                base_pix=self.base_pix,
            )

            for cul_metric in self.culuminative_metrics.keys():
                if "full_res" in cul_metric and "hp_masked" not in cul_metric:
                    self.culuminative_metrics[cul_metric].update(
                        pred,
                        full_res_mask,
                        cal_info,
                        hp_data=False,
                        foreground_pix=(gt_foreground, full_res_gt_foreground),
                        nside=self.nside,
                        base_pix=self.base_pix,
                    )

                elif "full_res" in cul_metric and "hp_masked" in cul_metric:
                    hp_masked_full_res = depth_utils.mask_flat_with_hp_cutout(
                        flat_data=full_res_mask.clone(),
                        cal_info=cal_info,
                        base_pix=self.base_pix,
                        nside=self.nside,
                        rotate_pole=self.rotate_pole,
                        masking_val=float("nan"),
                    )

                    hp_masked_pred = depth_utils.mask_flat_with_hp_cutout(
                        flat_data=pred[:, 0, ...].cpu().clone(),
                        cal_info=cal_info,
                        base_pix=self.base_pix,
                        nside=self.nside,
                        rotate_pole=self.rotate_pole,
                        masking_val=float("nan"),
                    ).to(
                        "cuda"
                    )  # Only compute chamfer dist on the hp masked preds
                    hp_masked_pred = hp_masked_pred[None, ...]

                    if self.mask_background:
                        hp_masked_full_res_foreground = depth_utils.get_foreground_mask(
                            hp_masked_full_res, background_val=(float("nan"), float("inf"), 1000)
                        )
                    else:
                        hp_masked_full_res_foreground = None

                    self.culuminative_metrics[cul_metric].update(
                        hp_masked_pred,
                        hp_masked_full_res,
                        cal_info,
                        hp_data=False,
                        foreground_pix=(gt_foreground, hp_masked_full_res_foreground),
                        nside=self.nside,
                        base_pix=self.base_pix,
                    )

                elif "small_res" in cul_metric and "hp_masked" in cul_metric:
                    small_res = self.small_res_transformation(full_res_mask.clone())
                    hp_masked_small_res = depth_utils.mask_flat_with_hp_cutout(
                        flat_data=small_res.clone(),
                        cal_info=cal_info,
                        base_pix=self.base_pix,
                        nside=self.nside,
                        rotate_pole=self.rotate_pole,
                        masking_val=float("nan"),
                    )
                    hp_masked_pred = depth_utils.mask_flat_with_hp_cutout(
                        flat_data=pred[:, 0, ...].cpu().clone(),
                        cal_info=cal_info,
                        base_pix=self.base_pix,
                        nside=self.nside,
                        rotate_pole=self.rotate_pole,
                        masking_val=float("nan"),
                    ).to(
                        "cuda"
                    )  # Only compute chamfer dist on the hp masked preds
                    hp_masked_pred = hp_masked_pred[None, ...]

                    if self.mask_background:
                        hp_masked_small_res_foreground = depth_utils.get_foreground_mask(
                            hp_masked_small_res, background_val=(float("nan"), float("inf"), 1000)
                        )
                    else:
                        hp_masked_small_res_foreground = None

                    self.culuminative_metrics[cul_metric].update(
                        hp_masked_pred,
                        hp_masked_small_res,
                        cal_info,
                        hp_data=False,
                        foreground_pix=(gt_foreground, hp_masked_small_res_foreground),
                        nside=self.nside,
                        base_pix=self.base_pix,
                    )

                else:
                    self.culuminative_metrics[cul_metric].update(
                        pred,
                        mask,
                        cal_info,
                        hp_data=False,
                        foreground_pix=gt_foreground,
                        nside=self.nside,
                        base_pix=self.base_pix,
                    )

            self.metric_values.append(metric_value.item())
            self.names.append(name)

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        metrics = {
            metric.replace("culum_", self.prefix): self.culuminative_metrics[metric].compute()
            for metric in self.culuminative_metrics.keys()
        }

        trainer.logger.log_metrics(metrics)

        self.metric_values = np.array(self.metric_values)
        self.names = np.array(self.names)

        sort_idcs = np.argsort(self.metric_values)
        if self.sort_dir == "desc":
            sort_idcs = sort_idcs[::-1]

        pred_names = {
            "top": self.names[sort_idcs[-self.top_k :]][::-1],
            "bottom": self.names[sort_idcs[: self.top_k]],
        }
        all_file_names = self.pred_dataset.hp_dataset.file_names

        for top_bottom, names in pred_names.items():
            print(f"writing predictions of {top_bottom} {self.top_k} samples...")
            for idx, name in enumerate(names):
                sample_idx = np.where(all_file_names == name + ".npz")[0].item()
                sample = self.pred_dataset[sample_idx]
                batch = sample["s2_img"].to(pl_module.device)[None, ...]
                cal_info = sample["cal_info"]
                preds = pl_module(batch)
                pred_mean = preds

                transf_mask = sample["s2_mask"][None, ...]
                transf_mask[transf_mask.isinf()] = float("nan")

                transf_mask = depth_utils.unnormalize_and_retransform(
                    data=transf_mask,
                    normalization=self.normalize_data,
                    data_stats=self.depth_data_statistics,
                    data_transform=self.data_transform,
                )

                metric_value = self.ranking_metric(
                    pred_mean,
                    transf_mask,
                    cal_info,
                    hp_data=False,
                    nside=self.nside,
                    base_pix=self.base_pix,
                )

                # ---------------- Generate plot of best and worst preds ----------------

                file_name_base = (
                    f"{self.prefix}{self.part}_{self.metric_name}_{top_bottom}_{idx+1}_{name}"
                )
                file_name_base += f"_{self.metric_name}={metric_value:.4f}"

                pred_pc, _ = depth_utils.create_point_cloud_from_depth_mask(
                    data=pred_mean[:, 0, ...],
                    cal_info=cal_info,
                    hp_data=False,
                    nside=self.nside,
                    base_pix=self.base_pix,
                )
                norm_pred_pc = torch.linalg.norm(pred_pc, dim=-1).flatten()
                gt_pc, gt_pc_fg = depth_utils.create_point_cloud_from_depth_mask(
                    data=transf_mask,
                    cal_info=cal_info,
                    hp_data=False,
                    background_val=(float("nan"), float("inf"), 1000),
                    nside=self.nside,
                    base_pix=self.base_pix,
                )
                norm_gt_pc = torch.linalg.norm(gt_pc, dim=-1).flatten()
                pred_pc = pred_pc.cpu()
                gt_pc = gt_pc.cpu()

                if self.mask_background:
                    pred_pc = pred_pc[:, gt_pc_fg.squeeze(), :]
                    gt_pc = gt_pc[:, gt_pc_fg.squeeze(), :]

                pred_norm_max = torch.amax(norm_pred_pc[~norm_pred_pc.isnan()]).to("cpu")
                gt_norm_max = torch.amax(norm_gt_pc[~norm_gt_pc.isnan()]).to("cpu")

                norm_pred_pc = norm_pred_pc.to("cpu")
                norm_gt_pc = norm_gt_pc.to("cpu")

                axes = ["x", "y", "z"]

                c_pred = np.tile(np.array([0.0, 0.0, 1.0]), (norm_pred_pc.shape[0], 1))
                c_pred_tmp = (
                    c_pred
                    * np.tile(np.expand_dims(norm_pred_pc.numpy(), -1), (1, 3))
                    / pred_norm_max.numpy()
                )
                c_pred -= c_pred_tmp
                pred_alpha = np.expand_dims(
                    1 - norm_pred_pc.numpy() / pred_norm_max.numpy(), axis=-1
                )
                c_pred = np.hstack([c_pred, pred_alpha])

                c_gt = np.tile(np.array([0.0, 1.0, 0.0]), (norm_gt_pc.shape[0], 1))
                c_gt_tmp = (
                    c_gt
                    * np.tile(np.expand_dims(norm_gt_pc.numpy(), -1), (1, 3))
                    / gt_norm_max.numpy()
                )
                c_gt -= c_gt_tmp
                gt_alpha = np.expand_dims(1 - norm_gt_pc.numpy() / gt_norm_max.numpy(), axis=-1)
                c_gt = np.hstack([c_gt, gt_alpha])

                if self.mask_background:
                    c_gt = c_gt[gt_pc_fg.squeeze(), :]
                    c_pred = c_pred[gt_pc_fg.squeeze(), :]

                for ind in range(3):
                    fig, ax = plt.subplots(figsize=(20, 20))

                    ax.scatter(
                        gt_pc[0, :, ind % 3], gt_pc[0, :, (ind + 1) % 3], s=1, c=c_gt, label="gt"
                    )
                    ax.scatter(
                        pred_pc[0, :, ind % 3],
                        pred_pc[0, :, (ind + 1) % 3],
                        s=1,
                        c=c_pred,
                        label="pred",
                    )
                    ax.set_title((axes[ind % 3] + axes[(ind + 1) % 3]).upper() + f" plane")
                    ax.set_xlabel(axes[ind % 3])
                    ax.set_ylabel(axes[(ind + 1) % 3])
                    ax.axis("equal")
                    ax.legend()
                    file_name = (
                        file_name_base
                        + f"_{(axes[ind % 3] + axes[(ind + 1) % 3]).upper()}_plane.png"
                    )
                    path = os.path.join(self.output_dir, file_name)
                    plt.savefig(path)
                    plt.close()

                small_res_img = torch.tensor(sample["s2_img"], dtype=torch.float32).permute(1, 2, 0)
                small_res_img = small_res_img.flatten(start_dim=0, end_dim=1)
                log_pred_pc, _ = depth_utils.create_point_cloud_from_depth_mask(
                    data=torch.log(pred_mean[:, 0, ...]),
                    cal_info=cal_info,
                    hp_data=False,
                    nside=self.nside,
                    base_pix=self.base_pix,
                )
                norm_log_pred_pc = torch.linalg.norm(log_pred_pc, dim=-1).flatten()
                log_gt_pc, _ = depth_utils.create_point_cloud_from_depth_mask(
                    data=torch.log(transf_mask),
                    cal_info=cal_info,
                    hp_data=False,
                    nside=self.nside,
                    base_pix=self.base_pix,
                )
                norm_log_gt_pc = torch.linalg.norm(log_gt_pc, dim=-1).flatten()
                log_pred_pc = log_pred_pc.cpu()
                log_gt_pc = log_gt_pc.cpu()

                norm_log_pred_pc = norm_log_pred_pc.to("cpu")
                norm_log_gt_pc = norm_log_gt_pc.to("cpu")

                c_img = small_res_img / torch.max(torch.linalg.norm(small_res_img, dim=-1))

                if self.mask_background:
                    c_img = c_img[gt_pc_fg.squeeze(), :]
                    log_pred_pc = log_pred_pc[:, gt_pc_fg.squeeze(), :]
                    log_gt_pc = log_gt_pc[:, gt_pc_fg.squeeze(), :]

                pred_mean = pred_mean.cpu()
                transf_mask = transf_mask.cpu()

                for ind in range(3):
                    fig, ax = plt.subplots(3, 2, figsize=(30, 20))

                    log_mask = ax[0, 0].imshow(torch.log(transf_mask.squeeze()))
                    ax[0, 0].set_title("log gt mask")
                    log_pred = ax[0, 1].imshow(torch.log(pred_mean.squeeze()))
                    ax[0, 1].set_title("log prediction")
                    plt.colorbar(log_mask, ax=ax[0, 0])
                    plt.colorbar(log_pred, ax=ax[0, 1])
                    ax[0, 0].set_axis_off()
                    ax[0, 1].set_axis_off()

                    ax[1, 0].scatter(
                        log_gt_pc[0, :, ind % 3],
                        log_gt_pc[0, :, (ind + 1) % 3],
                        s=1,
                        c=c_img,
                        label="gt",
                    )
                    ax[1, 0].set_title(
                        "log gt pc in " + (axes[ind % 3] + axes[(ind + 1) % 3]).upper() + f" plane"
                    )
                    ax[1, 0].set_xlabel(axes[ind % 3])
                    ax[1, 0].set_ylabel(axes[(ind + 1) % 3])
                    ax[1, 0].axis("equal")

                    ax[1, 1].scatter(
                        log_pred_pc[0, :, ind % 3],
                        log_pred_pc[0, :, (ind + 1) % 3],
                        s=1,
                        c=c_img,
                        label="pred",
                    )
                    ax[1, 1].set_title(
                        "log pred pc in "
                        + (axes[ind % 3] + axes[(ind + 1) % 3]).upper()
                        + f" plane"
                    )
                    ax[1, 1].set_xlabel(axes[ind % 3])
                    ax[1, 1].set_ylabel(axes[(ind + 1) % 3])
                    ax[1, 1].axis("equal")

                    ax[2, 0].scatter(
                        gt_pc[0, :, ind % 3], gt_pc[0, :, (ind + 1) % 3], s=1, c=c_img, label="gt"
                    )
                    ax[2, 0].set_title(
                        "gt pc in " + (axes[ind % 3] + axes[(ind + 1) % 3]).upper() + f" plane"
                    )
                    ax[2, 0].set_xlabel(axes[ind % 3])
                    ax[2, 0].set_ylabel(axes[(ind + 1) % 3])
                    ax[2, 0].axis("equal")

                    ax[2, 1].scatter(
                        pred_pc[0, :, ind % 3],
                        pred_pc[0, :, (ind + 1) % 3],
                        s=1,
                        c=c_img,
                        label="pred",
                    )
                    ax[2, 1].set_title(
                        "pred pc in " + (axes[ind % 3] + axes[(ind + 1) % 3]).upper() + f" plane"
                    )
                    ax[2, 1].set_xlabel(axes[ind % 3])
                    ax[2, 1].set_ylabel(axes[(ind + 1) % 3])
                    ax[2, 1].axis("equal")

                    file_name = (
                        file_name_base
                        + f"_{(axes[ind % 3] + axes[(ind + 1) % 3]).upper()}_plane_gt_vs_pred.png"
                    )
                    path = os.path.join(self.output_dir, file_name)
                    plt.savefig(path)
                    plt.close()
