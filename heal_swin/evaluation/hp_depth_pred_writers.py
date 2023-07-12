import os
import matplotlib.pyplot as plt
import copy

import numpy as np
import torch
import torchvision as tv
from pytorch_lightning.callbacks import BasePredictionWriter
import torchmetrics

from heal_swin.data.depth_estimation import normalize_depth_data
from heal_swin.utils import depth_utils
from heal_swin.evaluation import custom_metrics
from heal_swin.data.depth_estimation.project_depth_on_s2 import (
    project_depth_hp_img_back,
    project_depth_hp_mask_back,
)
from heal_swin.evaluation.flat_pred_writers import (
    BasePredictionWriterIgnoreArgs,
)


class WoodscapeHPDepthBasePredictionWriter(BasePredictionWriter):
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
        data_transform=None,
        mask_background=False,
        normalize_data=None,
        s2_bkgd_class=-1,
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
        self.s2_bkgd_class = s2_bkgd_class
        self.mask_background = mask_background
        self.data_transform = data_transform

        self.normalize_data = normalize_data

        self.depth_data_statistics = normalize_depth_data.get_depth_data_stats(
            data_transform=self.data_transform, mask_background=self.mask_background
        )

        self.mse = custom_metrics.DepthMSE()
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

            img_new = project_depth_hp_img_back(
                hp_img.numpy(),
                cal_info,
                self.resolution,
                self.rotate_pole,
                self.base_pix,
            )
            img_new = torch.tensor(img_new)

            hp_mask = depth_utils.unnormalize_and_retransform(
                data=hp_mask,
                normalization=self.normalize_data,
                data_stats=self.depth_data_statistics,
                data_transform=self.data_transform,
            )

            # pre_back_project_mse = self.mse(means[idxs], hp_mask[idxs])
            hp_pred = hp_pred.unsqueeze(0)
            hp_mask = hp_mask.unsqueeze(0)
            pre_back_project_mse = self.mse(hp_pred, hp_mask)

            means = hp_pred  # [:,0, ...]
            means = (
                torch.tensor(
                    project_depth_hp_mask_back(
                        hp_mask=means.squeeze(),
                        cal_info=cal_info,
                        output_resolution=self.resolution,
                        rotate_pole=self.rotate_pole,
                        nside=self.nside,
                        base_pix=self.base_pix,
                        s2_bkgd_class=float("nan"),
                    )
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            pred_mean = means

            transf_mask = torch.tensor(
                project_depth_hp_mask_back(
                    hp_mask=hp_mask.squeeze(),  # [0,...],
                    cal_info=cal_info,
                    output_resolution=self.resolution,
                    rotate_pole=self.rotate_pole,
                    nside=self.nside,
                    base_pix=self.base_pix,
                    s2_bkgd_class=float("nan"),
                )
            ).unsqueeze(0)
            # transf_mask = torch.from_numpy(transf_mask)
            transf_mask[transf_mask.isinf()] = float("nan")

            if pl_module.use_logvar:
                std = torch.sqrt(torch.exp(hp_pred[:, 1, ...]))
                std = std.numpy()
                std = project_depth_hp_mask_back(
                    hp_mask=std,
                    cal_info=cal_info,
                    output_resolution=self.resolution,
                    rotate_pole=self.rotate_pole,
                    nside=self.nside,
                    base_pix=self.base_pix,
                    s2_bkgd_class=float("nan"),
                )
                std = torch.tensor(std)
                std = torch.squeeze(std)
            else:
                std = None

            post_back_project_mse = self.mse(pred_mean, transf_mask)
            ranged_mse_map = depth_utils.create_ranged_mse_mask(
                pl_module.metric_dict, pred_mean, transf_mask
            ).squeeze()

            original_size = mask.shape[-2:]
            output_size = tuple([int(dim * self.resolution) for dim in original_size])
            transform = tv.transforms.Resize(
                output_size, interpolation=self.interpolation_modes[self.interpolation_mode]
            )  # Scale the image to self.resolution * original size
            img_transform = tv.transforms.Resize(
                output_size, interpolation=self.interpolation_modes[self.interpolation_mode]
            )

            pred_mean = pred_mean.squeeze()
            pred_mean = transform(pred_mean.unsqueeze(0)).squeeze()
            transf_mask = transform(transf_mask).squeeze()
            img = img_transform(img)
            mask = transform(mask.unsqueeze(0)).squeeze()
            mask[mask.isinf()] = float("nan")

            if std is not None:
                std = transform(std)

            plt_path = os.path.join(
                self.output_dir,
                f"{self.prefix}{self.part}_{name}_best_ckpt_mse={pre_back_project_mse:.4f}_"
                f"{post_back_project_mse:.4f}.png",
            )
            depth_utils.save_depth_pred_comparison_image(
                pred_dist=pred_mean,
                original_ground_truth=mask,
                ground_truth_after_transforms=transf_mask,
                img=img,
                filepath=plt_path,
                metric_dict=pl_module.metric_dict,
                std=std,
                ranged_mse_map=ranged_mse_map,
            )


class WoodscapeHPDepthChamferDistancePredictionWriter(BasePredictionWriter):
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
        data_transform=None,
        mask_background=False,
        normalize_data=None,
        s2_bkgd_class=-1,
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
        self.s2_bkgd_class = s2_bkgd_class
        self.mask_background = mask_background
        self.data_transform = data_transform

        self.normalize_data = normalize_data

        self.depth_data_statistics = normalize_depth_data.get_depth_data_stats(
            data_transform=self.data_transform, mask_background=self.mask_background
        )

        self.mse = torchmetrics.MeanSquaredError()
        self.mse = custom_metrics.DepthMSE()
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

            img_new = project_depth_hp_img_back(
                hp_img.numpy(),
                cal_info,
                self.resolution,
                self.rotate_pole,
                self.base_pix,
            )
            img_new = torch.tensor(img_new)

            means = hp_pred[0, :]

            hp_mask = depth_utils.unnormalize_and_retransform(
                data=hp_mask,
                normalization=self.normalize_data,
                data_stats=self.depth_data_statistics,
                data_transform=self.data_transform,
            )

            idxs = ~hp_mask.isnan()

            pre_back_project_mse = self.mse(means[idxs], hp_mask[idxs])

            means = project_depth_hp_mask_back(
                hp_mask=means,
                cal_info=cal_info,
                output_resolution=self.resolution,
                rotate_pole=self.rotate_pole,
                nside=self.nside,
                base_pix=self.base_pix,
                s2_bkgd_class=float("nan"),
            )
            pred_mean = torch.from_numpy(means)

            transf_mask = project_depth_hp_mask_back(
                hp_mask=hp_mask,
                cal_info=cal_info,
                output_resolution=self.resolution,
                rotate_pole=self.rotate_pole,
                nside=self.nside,
                base_pix=self.base_pix,
                s2_bkgd_class=float("nan"),
            )
            transf_mask = torch.from_numpy(transf_mask)
            transf_mask[transf_mask.isinf()] = float("nan")

            if pl_module.use_logvar:
                std = torch.sqrt(torch.exp(hp_pred[1, :]))
                std = std.numpy()
                std = project_depth_hp_mask_back(
                    hp_mask=std,
                    cal_info=cal_info,
                    output_resolution=self.resolution,
                    rotate_pole=self.rotate_pole,
                    nside=self.nside,
                    base_pix=self.base_pix,
                    s2_bkgd_class=float("nan"),
                )
                std = torch.tensor(std)
                std = torch.squeeze(std)
            else:
                std = None

            post_back_project_mse = self.mse(
                pred_mean[~transf_mask.isnan()], transf_mask[~transf_mask.isnan()]
            )
            ranged_mse_map = depth_utils.create_ranged_mse_mask(
                pl_module.metric_dict, pred_mean, transf_mask
            ).squeeze()

            original_size = mask.shape[-2:]
            output_size = tuple([int(dim * self.resolution) for dim in original_size])
            transform = tv.transforms.Resize(
                output_size, interpolation=self.interpolation_modes[self.interpolation_mode]
            )  # Scale the image to self.resolution * original size
            img_transform = tv.transforms.Resize(
                output_size, interpolation=self.interpolation_modes[self.interpolation_mode]
            )

            pred_mean = transform(pred_mean).squeeze()
            transf_mask = transform(transf_mask).squeeze()
            img = img_transform(img)
            mask = transform(mask.unsqueeze(0)).squeeze()
            mask[mask.isinf()] = float("nan")

            if std is not None:
                std = transform(std)

            idxs = ~transf_mask.isnan()

            plt_path = os.path.join(
                self.output_dir,
                f"{self.prefix}{self.part}_{name}_best_ckpt_mse={pre_back_project_mse:.4f}_"
                f"{post_back_project_mse:.4f}.png",
            )
            depth_utils.save_depth_pred_comparison_image(
                pred_dist=pred_mean,
                original_ground_truth=mask,
                ground_truth_after_transforms=transf_mask,
                img=img,
                filepath=plt_path,
                metric_dict=pl_module.metric_dict,
                std=std,
                ranged_mse_map=ranged_mse_map,
            )


class WoodscapeHPDepthValOnBackProjectedPredictionWriter(BasePredictionWriterIgnoreArgs):
    def __init__(
        self,
        output_dir,
        write_interval,
        img_dims,
        rotate_pole=False,
        prefix="",
        nside=256,
        base_pix=8,
        part="val",
        ignore_gt_classes=[],
        woodscape_version=None,
        proj_res=1,
        data_transform=False,
        mask_background=False,
        normalize_data=None,
        interpolation_mode="nearest",
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
        self.nside = nside
        self.base_pix = base_pix
        self.part = part
        self.proj_res = proj_res
        self.org_size = None

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

        self.proj_res = proj_res
        # self.mask_transform = tv.transforms.Resize(
        #     proj_res, interpolation=InterpolationMode.BILINEAR
        # )  # proj_res is a float, how does this make sense?

        # It should only be an int, and if it is an int:
        # If size is an int, smaller edge of the image will be matched to this
        # number. i.e, if height > width, then image will be rescaled to
        # (size * height / width, size).
        # So proj_res should really be proj_res * smallest side length

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
        assert (
            pl_module.global_rank == 0
        ), "currently the back-projected validation works only on a single GPU"
        hp_preds = prediction  # This should be non-normalized, non-transformed data
        hp_preds = hp_preds.cpu()
        masks = batch["masks"]  # This should be non-normalizes, non-transformed data

        cal_infos = batch["cal_infos"]
        if self.org_size is None:
            cal_info = cal_infos[0]["intrinsic"]
            self.org_size = int(cal_info["height"]), int(cal_info["width"])

        flat_preds = []
        for idx, (hp_pred, mask, cal_info) in enumerate(zip(hp_preds, masks, cal_infos)):
            pred_mean = hp_pred[0, ...]
            pred_mean = torch.tensor(
                project_depth_hp_mask_back(
                    pred_mean.numpy(),
                    cal_info,
                    output_resolution=self.proj_res,
                    rotate_pole=self.rotate_pole,
                    nside=self.nside,
                    base_pix=self.base_pix,
                    s2_bkgd_class=float("nan"),
                )
            )  # .squeeze()
            if pl_module.use_logvar:
                pred_std = hp_pred[1, ...]
                pred_std = torch.tensor(
                    project_depth_hp_mask_back(
                        pred_std.numpy(),
                        cal_info,
                        output_resolution=self.proj_res,
                        rotate_pole=self.rotate_pole,
                        nside=self.nside,
                        base_pix=self.base_pix,
                        s2_bkgd_class=float("nan"),
                    )
                )  # .squeeze()

            else:
                pred_std = None

            flat_preds.append(pred_mean)
            if isinstance(self.proj_res, (float, int)):
                output = int(self.proj_res * min(mask.shape))
            else:
                output = self.proj_res

            mask_transform = tv.transforms.Resize(
                output, interpolation=self.interpolation_modes[self.interpolation_mode]
            )
            mask = mask_transform(mask.unsqueeze(0)).squeeze()
            pred_mean = mask_transform(
                pred_mean.unsqueeze(0)
            ).squeeze()  # should return a [h,w] map
            if pred_std is not None:
                pred_std = mask_transform(
                    pred_std.unsqueeze(0)
                ).squeeze()  # should return a [h,w] map
                preds = torch.stack([pred_mean, pred_std], axis=0)
            else:
                preds = torch.stack([pred_mean], axis=0)

            preds = preds.unsqueeze(0)
            mask = mask.unsqueeze(0)

            for met_key in self.metric_dict:
                self.metric_dict[met_key].update(
                    preds.clone(), mask.clone()
                )  # These dimensions could be problematic

        flat_pred_batch = torch.cat(flat_preds, dim=0)
        return flat_pred_batch, masks

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        pref = f"{self.prefix}{self.part}"
        suff = "back_projected"

        if isinstance(self.proj_res, int) and self.proj_res != min(self.org_size):
            suff += f"_res_{self.proj_res}"
        elif isinstance(self.proj_res, tuple):
            suff += f"_res_{self.proj_res[0]}_{self.proj_res[1]}"

        metrics = {}
        for met_key in self.metric_dict:
            metrics[pref + "_" + met_key + "_" + suff] = self.metric_dict[met_key].compute().item()
        trainer.logger.log_metrics(metrics)


class WoodscapeHPDepthBestWorstPredictionWriter(BasePredictionWriter):
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
        **_ignore,
    ):
        # This works
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
        hp_preds = prediction  # This should be non-normalized and non-transformed as per output
        hp_preds = hp_preds.cpu()
        hp_masks = batch["hp_masks"].cpu()  # This should be normalized and transformed
        names = batch["names"]

        for hp_pred, hp_mask, name in zip(hp_preds, hp_masks, names):
            hp_mask = depth_utils.unnormalize_and_retransform(
                data=hp_mask,
                normalization=self.normalize_data,
                data_stats=self.depth_data_statistics,
                data_transform=self.data_transform,
            )

            hp_mask = hp_mask.unsqueeze(0)
            hp_pred = hp_pred.unsqueeze(0)

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
                batch = sample["hp_img"].to(pl_module.device)[None, ...]
                hp_pred = pl_module(batch)  # Get the model output, non-normalized, non-transformed
                # This is probably wrong with respect to the output channels
                hp_pred = hp_pred.cpu()  # .squeeze().cpu()

                hp_mask = sample["hp_mask"]

                hp_mask = depth_utils.unnormalize_and_retransform(
                    data=hp_mask,
                    normalization=self.normalize_data,
                    data_stats=self.depth_data_statistics,
                    data_transform=self.data_transform,
                )

                metric_value = self.ranking_metric(hp_pred, hp_mask)

                img_new = project_depth_hp_img_back(
                    sample["hp_img"],
                    sample["cal_info"],
                    self.resolution,
                    self.rotate_pole,
                    self.base_pix,
                )
                img_new = torch.tensor(img_new, dtype=torch.uint8)
                pred_mean = torch.tensor(
                    project_depth_hp_mask_back(
                        hp_mask=hp_pred.squeeze().numpy(),
                        cal_info=sample["cal_info"],
                        output_resolution=self.resolution,
                        rotate_pole=self.rotate_pole,
                        nside=self.nside,
                        base_pix=self.base_pix,
                        s2_bkgd_class=float("nan"),
                    )
                )
                transf_mask = torch.tensor(
                    project_depth_hp_mask_back(
                        hp_mask=hp_mask.squeeze(),
                        cal_info=sample["cal_info"],
                        output_resolution=self.resolution,
                        rotate_pole=self.rotate_pole,
                        nside=self.nside,
                        base_pix=self.base_pix,
                        s2_bkgd_class=float("nan"),
                    )
                )
                proj_metric_value = self.ranking_metric(
                    pred_mean.unsqueeze(0).unsqueeze(0), transf_mask.unsqueeze(0)
                )

                output_size = sample["mask"].shape[-2:]
                gt_overlay_transform = tv.transforms.Resize(output_size)
                transf_mask = gt_overlay_transform(transf_mask).squeeze()
                transf_mask[transf_mask.isinf()] = float("nan")
                pred_mean = gt_overlay_transform(pred_mean).squeeze()

                file_name = (
                    f"{self.prefix}{self.part}_{self.metric_name}_{top_bottom}_{idx+1}_{name}"
                )
                file_name += f"_{self.metric_name}={metric_value:.4f}_{proj_metric_value:.4f}.png"
                path = os.path.join(self.output_dir, file_name)

                depth_utils.save_depth_pred_comparison_image(
                    pred_dist=pred_mean,
                    original_ground_truth=sample["mask"],
                    ground_truth_after_transforms=transf_mask,
                    img=img_new,
                    filepath=path,
                    metric_dict=pl_module.metric_dict,
                    std=None,
                )


class WoodscapeHPDepthChamferDistBestWorstPredictionWriter(BasePredictionWriter):
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
        # self.culuminative_ranking_metric = copy.deepcopy(self.ranking_metric)
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
        # So this can be called both on val and pred batch end?
        hp_preds = prediction  # This should be non-normalized and non-transformed as per output
        hp_preds = hp_preds.cpu()
        hp_masks = batch["hp_masks"].cpu()  # This should be normalized and transformed
        full_res_masks = batch["masks"]
        names = batch["names"]
        cal_infos = batch["cal_infos"]

        # This gets the wrong indices
        for ind, (hp_pred, hp_mask, name, cal_info, full_res_mask) in enumerate(
            zip(hp_preds, hp_masks, names, cal_infos, full_res_masks)
        ):

            hp_mask = depth_utils.unnormalize_and_retransform(
                data=hp_mask,
                normalization=self.normalize_data,
                data_stats=self.depth_data_statistics,
                data_transform=self.data_transform,
            )

            hp_mask = hp_mask[None, ...]  # This should result in a shape [1, ind]
            hp_pred = hp_pred[None, ...]  # This should result in a shape [1, c_out, ind]
            hp_mask[hp_mask.isinf()] = float("nan")

            full_res_mask = full_res_mask[None, ...]
            if self.mask_background:
                hp_foreground = depth_utils.get_foreground_mask(
                    hp_mask, background_val=(float("nan"), float("inf"), 1000)
                )
                full_res_gt_foreground = depth_utils.get_foreground_mask(
                    full_res_mask, background_val=(float("nan"), float("inf"), 1000)
                )
            else:
                hp_foreground = None
                full_res_gt_foreground = None

            metric_value = self.ranking_metric(
                hp_pred,
                hp_mask,
                cal_info,
                hp_data=True,
                foreground_pix=hp_foreground,
                nside=self.nside,
                base_pix=self.base_pix,
            )

            for cul_metric in self.culuminative_metrics.keys():

                if "full_res" in cul_metric and "hp_masked" not in cul_metric:
                    self.culuminative_metrics[cul_metric].update(
                        hp_pred,
                        full_res_mask,
                        cal_info,
                        hp_data=(True, False),
                        foreground_pix=(hp_foreground, full_res_gt_foreground),
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

                    if self.mask_background:
                        hp_masked_full_res_foreground = depth_utils.get_foreground_mask(
                            hp_masked_full_res, background_val=(float("nan"), float("inf"), 1000)
                        )
                    else:
                        hp_masked_full_res_foreground = None

                    self.culuminative_metrics[cul_metric].update(
                        hp_pred,
                        hp_masked_full_res,
                        cal_info,
                        hp_data=(True, False),
                        foreground_pix=(hp_foreground, hp_masked_full_res_foreground),
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

                    if self.mask_background:
                        hp_masked_small_res_foreground = depth_utils.get_foreground_mask(
                            hp_masked_small_res, background_val=(float("nan"), float("inf"), 1000)
                        )
                    else:
                        hp_masked_small_res_foreground = None

                    self.culuminative_metrics[cul_metric].update(
                        hp_pred,
                        hp_masked_small_res,
                        cal_info,
                        hp_data=(True, False),
                        foreground_pix=(hp_foreground, hp_masked_small_res_foreground),
                        nside=self.nside,
                        base_pix=self.base_pix,
                    )

                else:
                    self.culuminative_metrics[cul_metric].update(
                        hp_pred,
                        hp_mask,
                        cal_info,
                        hp_data=True,
                        foreground_pix=hp_foreground,
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
        all_file_names = self.pred_dataset.hp_imgs_masks_dataset.file_names

        for top_bottom, names in pred_names.items():
            print(f"writing predictions of {top_bottom} {self.top_k} samples...")
            for idx, name in enumerate(names):
                sample_idx = np.where(all_file_names == name + ".npz")[0].item()
                sample = self.pred_dataset[sample_idx]
                batch = sample["hp_img"].to(pl_module.device)[None, ...]
                cal_info = sample["cal_info"]
                hp_pred = pl_module(batch)  # Get the model output, non-normalized, non-transformed
                # This is probably wrong with respect to the output channels

                hp_mask = sample["hp_mask"][None, ...]
                hp_mask = depth_utils.unnormalize_and_retransform(
                    data=hp_mask,
                    normalization=self.normalize_data,
                    data_stats=self.depth_data_statistics,
                    data_transform=self.data_transform,
                )

                if self.mask_background:
                    hp_foreground = depth_utils.get_foreground_mask(
                        hp_mask, background_val=(float("nan"), float("inf"), 1000)
                    )
                else:
                    hp_foreground = None

                metric_value = self.ranking_metric(
                    hp_pred,
                    hp_mask,
                    cal_info,
                    hp_data=True,
                    foreground_pix=hp_foreground,
                    nside=self.nside,
                    base_pix=self.base_pix,
                )

                # ------------- Generate plot of best and worst preds -------------

                file_name_base = (
                    f"{self.prefix}{self.part}_{self.metric_name}_{top_bottom}_{idx+1}_{name}"
                )
                file_name_base += f"_{self.metric_name}={metric_value:.4f}"

                # ------------- Generate pc from hp pred and gt -------------
                pred_pc, _ = depth_utils.create_point_cloud_from_depth_mask(
                    data=hp_pred[:, 0, ...],
                    cal_info=cal_info,
                    hp_data=True,
                    nside=self.nside,
                    base_pix=self.base_pix,
                )
                norm_pred_pc = torch.linalg.norm(pred_pc, dim=-1).flatten()
                gt_pc, gt_pc_fg = depth_utils.create_point_cloud_from_depth_mask(
                    data=hp_mask,
                    cal_info=cal_info,
                    hp_data=True,
                    background_val=(float("nan"), float("inf"), 1000),
                    nside=self.nside,
                    base_pix=self.base_pix,
                )
                norm_gt_pc = torch.linalg.norm(gt_pc, dim=-1).flatten()

                if self.mask_background:
                    pred_pc = pred_pc[:, gt_pc_fg.squeeze(), :]
                    gt_pc = gt_pc[:, gt_pc_fg.squeeze(), :]

                pred_pc = pred_pc.cpu()
                gt_pc = gt_pc.cpu()

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
                        gt_pc[0, :, ind % 3], gt_pc[0, :, (ind + 1) % 3], s=1, c=c_gt, label="hp gt"
                    )
                    ax.scatter(
                        pred_pc[0, :, ind % 3],
                        pred_pc[0, :, (ind + 1) % 3],
                        s=1,
                        c=c_pred,
                        label="hp pred",
                    )
                    ax.set_title((axes[ind % 3] + axes[(ind + 1) % 3]).upper() + " plane")
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

                small_res_img = torch.tensor(sample["hp_img"], dtype=torch.float32)

                log_pred_pc, log_pred_pc_fg = depth_utils.create_point_cloud_from_depth_mask(
                    data=torch.log(hp_pred[:, 0, ...]),
                    cal_info=cal_info,
                    hp_data=True,
                    nside=self.nside,
                    base_pix=self.base_pix,
                )

                log_gt_pc, log_gt_pc_fg = depth_utils.create_point_cloud_from_depth_mask(
                    data=torch.log(hp_mask),
                    cal_info=cal_info,
                    hp_data=True,
                    nside=self.nside,
                    base_pix=self.base_pix,
                )

                log_pred_pc = log_pred_pc.cpu()
                log_gt_pc = log_gt_pc.cpu()

                c_img = small_res_img / 255
                c_img = c_img.permute((1, 0))

                hp_pred = hp_pred.cpu()  # Should be of shape [1,c_out, ind]
                hp_pred = hp_pred[:, 0, ...]
                hp_mask = hp_mask.cpu()

                back_proj_hp_pred = torch.tensor(
                    project_depth_hp_mask_back(
                        hp_mask=hp_pred.squeeze(),
                        cal_info=sample["cal_info"],
                        output_resolution=self.resolution,
                        rotate_pole=self.rotate_pole,
                        nside=self.nside,
                        base_pix=self.base_pix,
                        s2_bkgd_class=float("nan"),
                    )
                )
                back_proj_hp_mask = torch.tensor(
                    project_depth_hp_mask_back(
                        hp_mask=hp_mask.squeeze(),
                        cal_info=sample["cal_info"],
                        output_resolution=self.resolution,
                        rotate_pole=self.rotate_pole,
                        nside=self.nside,
                        base_pix=self.base_pix,
                        s2_bkgd_class=float("nan"),
                    )
                )

                if self.mask_background:
                    c_img = c_img[gt_pc_fg.squeeze(), :]
                    log_pred_pc = log_pred_pc[:, gt_pc_fg.squeeze(), :]
                    log_gt_pc = log_gt_pc[:, gt_pc_fg.squeeze(), :]

                for ind in range(3):
                    fig, ax = plt.subplots(3, 2, figsize=(30, 20))

                    log_mask = ax[0, 0].imshow(torch.log(back_proj_hp_mask.squeeze()))
                    ax[0, 0].set_title("log gt mask")
                    log_pred = ax[0, 1].imshow(torch.log(back_proj_hp_pred.squeeze()))
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
                        "log gt pc in " + (axes[ind % 3] + axes[(ind + 1) % 3]).upper() + " plane"
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
                        "log pred pc in " + (axes[ind % 3] + axes[(ind + 1) % 3]).upper() + " plane"
                    )
                    ax[1, 1].set_xlabel(axes[ind % 3])
                    ax[1, 1].set_ylabel(axes[(ind + 1) % 3])
                    ax[1, 1].axis("equal")

                    ax[2, 0].scatter(
                        gt_pc[0, :, ind % 3], gt_pc[0, :, (ind + 1) % 3], s=1, c=c_img, label="gt"
                    )
                    ax[2, 0].set_title(
                        "gt pc in " + (axes[ind % 3] + axes[(ind + 1) % 3]).upper() + " plane"
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
                        "pred pc in " + (axes[ind % 3] + axes[(ind + 1) % 3]).upper() + " plane"
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
