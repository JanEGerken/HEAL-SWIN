from typing import Dict, Callable, List, Union, Tuple

import numpy as np
import torch
from torchmetrics import Metric, IoU
import chamfer_distance
import torchvision as tv

from heal_swin.data.segmentation.project_on_s2 import project_hp_mask_back


def get_non_inf_non_nan_idxs(tens1: torch.Tensor, tens2: torch.Tensor) -> torch.Tensor:
    pred_inf = tens1.isinf()
    gt_inf = tens2.isinf()
    non_inf_idxs = ~(pred_inf | gt_inf)

    pred_nan = tens1.isnan()
    gt_nan = tens2.isnan()
    non_nan_idxs = ~(pred_nan | gt_nan)

    idxs = non_inf_idxs & non_nan_idxs
    return idxs


class HPMaskedIoU(Metric):
    def __init__(
        self,
        cal_info,
        nside,
        base_pix,
        rotate_pole,
        proj_res,
        num_classes,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.mask = self.get_mask(cal_info, nside, base_pix, rotate_pole, proj_res)
        self.iou = IoU(num_classes=num_classes, reduction="none")

    def get_mask(self, cal_info, nside, base_pix, rotate_pole, proj_res):
        hp_seg_mask = torch.zeros((base_pix * nside**2), dtype=torch.uint8)
        flat_seg_mask = project_hp_mask_back(
            hp_seg_mask, cal_info, proj_res, rotate_pole, nside, base_pix, s2_bkgd_class=1
        )
        return flat_seg_mask == 0

    def update(self, preds, target):
        self.iou.update(preds[:, self.mask], target[:, self.mask])

    def compute(self):
        return self.iou.compute()


class DepthMSE(Metric):
    def __init__(
        self,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("num_elements", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("sum_se", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        means = preds[
            :, 0, ...
        ]  # output shape should be [N, H, W] to match target with supposed same shape
        means = means.squeeze()
        target = target.squeeze()
        assert means.shape == target.shape, (
            "the predicted mean and target need to have the same shape, "
            f"got means {means.shape} and target {target.shape}"
        )

        idxs = get_non_inf_non_nan_idxs(means, target)

        sq_diff = torch.square(means[idxs] - target[idxs])

        self.sum_se += torch.sum(sq_diff)
        num_elem = torch.count_nonzero(idxs)
        self.num_elements += num_elem

    def compute(self) -> torch.Tensor:
        return self.sum_se / self.num_elements


class DepthRelSE(Metric):
    def __init__(
        self,
        total_mean,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.total_mean = total_mean
        self.add_state("sum_pred_se", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("sum_mean_se", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        means = preds[:, 0, ...]

        idxs = get_non_inf_non_nan_idxs(means, target)

        sq_pred_diff = torch.square(means[idxs] - target[idxs])
        sq_mean_diff = torch.square(self.total_mean - target[idxs])

        self.sum_pred_se += torch.sum(sq_pred_diff)
        self.sum_mean_se += torch.sum(sq_mean_diff)

    def compute(self) -> torch.Tensor:
        return self.sum_pred_se / self.sum_mean_se


class DepthRelAE(Metric):
    def __init__(
        self,
        total_mean,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.total_mean = total_mean
        self.add_state("sum_pred_ae", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("sum_mean_ae", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        means = preds[:, 0, ...]

        idxs = get_non_inf_non_nan_idxs(means, target)

        sq_pred_diff = torch.abs(means[idxs] - target[idxs])
        sq_mean_diff = torch.abs(self.total_mean - target[idxs])

        self.sum_pred_ae += torch.sum(sq_pred_diff)
        self.sum_mean_ae += torch.sum(sq_mean_diff)

    def compute(self) -> torch.Tensor:
        return self.sum_pred_ae / self.sum_mean_ae


class DepthiRMSE(Metric):
    def __init__(
        self,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("num_elements", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("sum_inv_se", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        means = preds[:, 0, ...]

        means = means.clone()
        target = target.clone()

        means = 1 / (0.001 * means)  # to 1/km
        target = 1 / (0.001 * target)

        idxs = get_non_inf_non_nan_idxs(means, target)

        sq_diff = torch.square(means[idxs] - target[idxs])

        self.sum_inv_se += torch.sum(sq_diff)
        num_elem = torch.count_nonzero(idxs)
        self.num_elements += num_elem

    def compute(self) -> torch.Tensor:
        return torch.sqrt(self.sum_inv_se / self.num_elements)


class DepthRangeMSE(Metric):
    def __init__(
        self,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
        distance_range=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("num_elements", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("sum_se", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

        if isinstance(distance_range, (tuple, list)):
            if len(distance_range) == 1:
                distance_range = distance_range[0]  # Extract the provided value
            else:
                assert (
                    len(distance_range) == 2
                ), f"Range needs to be two numbers, got distance_range={distance_range}..."
                self.min = min(distance_range)
                self.max = max(distance_range)

        if isinstance(distance_range, (int, float)):
            self.min = float("-inf")
            self.max = distance_range

        self.distance_range = [self.min, self.max]

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        means = preds[:, 0, ...]

        idxs = get_non_inf_non_nan_idxs(means, target)

        sq_diff = torch.square(means - target)

        relevant_inds = (
            (self.distance_range[0] <= target) & (target < self.distance_range[1]) & idxs
        )

        self.sum_se += torch.sum(sq_diff[relevant_inds])
        num_elem = torch.count_nonzero(relevant_inds)
        self.num_elements += num_elem

    def compute(self) -> torch.Tensor:
        if self.num_elements == 0:
            return 0  # No ground truth in range
        else:
            return self.sum_se / self.num_elements


def add_distance_ranged_mse(
    metric_dict: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    distance_ranges: List[Union[Tuple[int], int, float]],
) -> Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    max_digits = 0
    flat = np.array(distance_ranges).flatten()
    for num in flat:
        if len(str(num)) > max_digits:
            max_digits = len(str(num))

    for ran in distance_ranges:
        if len(ran) == 2:
            min_range, max_range = ran
            min_range = str(min_range)
            max_range = str(max_range)

            min_range = f"{min_range:0>{max_digits}}"

            max_range = f"{max_range:0>{max_digits}}"

            range_str = min_range + "_" + max_range

        elif len(ran) == 1:
            distance_range = ran[0]
            range_str = "_neg_inf_" + str(distance_range)

        elif isinstance(ran, (int, float)):
            distance_range = ran
            range_str = "_neg_inf_" + str(distance_range)

        metric_dict["mse_range_" + range_str] = DepthRangeMSE(distance_range=ran)

    return metric_dict


class MeanSTD(Metric):
    def __init__(
        self,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("num_elements", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("sum_stds", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        tmp_target = target.clone()
        tmp_target[tmp_target == float("inf")] = float("nan")
        idces = ~tmp_target.isnan()
        log_var = preds[:, 1, ...]
        self.sum_stds += torch.sum(torch.sqrt(torch.exp(log_var[idces])))
        self.num_elements += torch.count_nonzero(idces)

    def compute(self) -> torch.Tensor:
        return self.sum_stds / self.num_elements


class MeanSTDMedian(Metric):
    def __init__(
        self,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("sum_median", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("num_elements", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        log_var = preds[:, 1, ...]
        N = log_var.shape[0]
        stds = torch.sqrt(torch.exp(log_var))
        stds_median = stds.view(N, -1).median(1).values
        self.sum_median += torch.sum(stds_median)
        self.num_elements += stds_median.numel()

    def compute(self) -> torch.Tensor:
        return self.sum_median / self.num_elements


class MeanPredDist(Metric):
    def __init__(
        self,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("sum_pred_dist", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("num_elements", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        pred_dist = preds[:, 0, ...]

        idxs = get_non_inf_non_nan_idxs(pred_dist, target)

        self.sum_pred_dist += torch.sum(pred_dist[idxs])
        self.num_elements += pred_dist[idxs].numel()

    def compute(self) -> torch.Tensor:
        return self.sum_pred_dist / self.num_elements


class STDPredDist(Metric):
    def __init__(
        self,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("M2", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("num_samples", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("running_mean", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        pred_dist = preds[:, 0, ...]

        idxs = get_non_inf_non_nan_idxs(pred_dist, target)
        prev_samples = self.num_samples.clone()
        self.num_samples += pred_dist[idxs].numel()
        delta = torch.mean(pred_dist[idxs]) - self.running_mean
        self.running_mean = (
            self.running_mean * prev_samples / self.num_samples
            + torch.mean(pred_dist[idxs]) / self.num_samples
        )
        delta2 = torch.mean(pred_dist[idxs]) - self.running_mean
        self.M2 += delta * delta2

    def compute(self) -> torch.Tensor:
        if self.num_samples < 2:
            return float("nan")
        else:
            variance = self.M2 / self.num_sampels
            return torch.sqrt(variance)


class ScaleInvariantLogError(Metric):
    """https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction"""

    def __init__(
        self,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("sum_d_sq", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("sum_d", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("num_elements", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        pred_dist = preds[:, 0, ...]

        idxs = get_non_inf_non_nan_idxs(pred_dist, target)
        idxs = idxs & (pred_dist > 0)
        idxs = idxs & (target > 0)

        diff_log = torch.log(target[idxs]) - torch.log(pred_dist[idxs])

        num_elem = diff_log.numel()

        self.sum_d_sq += torch.sum(torch.square(diff_log))
        self.sum_d += torch.sum(diff_log)
        self.num_elements += num_elem

    def compute(self) -> torch.Tensor:
        siloge = 1 / self.num_elements * self.sum_d_sq - 1 / torch.square(
            self.num_elements
        ) * torch.square(self.sum_d)
        return siloge


class ChamferDistance(Metric):
    def __init__(
        self,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.chamfer_distance_module = chamfer_distance.ChamferDistance()
        self.add_state("sum_chamfer", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("num_samples", torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(
        self,
        preds,
        target,
        cal_info,
        nside=256,
        base_pix=8,
        hp_data=False,
        rotate_pole=False,
        foreground_pix=None,
    ):
        """preds: tensor of shape [N, 1/2, H, W] for "flat data", [N, 1/2, ind] for hp data
        target: tensor of shape [N, H, W] for "flat data", [N, ind] for hp data
        """
        from heal_swin.utils import depth_utils

        preds.to("cuda")
        target.to("cuda")

        if isinstance(hp_data, (list, tuple)):
            hp_data_pred = hp_data[0]
            hp_data_target = hp_data[1]
        else:
            hp_data_pred = hp_data
            hp_data_target = hp_data

        pred_dist = preds[:, 0, ...]
        pred_shape = pred_dist.shape
        target_shape = target.shape

        pred_pc, _ = depth_utils.create_point_cloud_from_depth_mask(
            data=pred_dist,
            cal_info=cal_info,
            nside=nside,
            base_pix=base_pix,
            hp_data=hp_data_pred,
            rotate_pole=rotate_pole,
        )
        target_pc, _ = depth_utils.create_point_cloud_from_depth_mask(
            data=target,
            cal_info=cal_info,
            nside=nside,
            base_pix=base_pix,
            hp_data=hp_data_target,
            rotate_pole=rotate_pole,
        )

        pred_pc = pred_pc.to("cuda")
        target_pc = target_pc.to("cuda")

        tmp_sum_pred = torch.sum(pred_pc, dim=-1)
        tmp_sum_target = torch.sum(target_pc, dim=-1)

        non_nan_pred_inds = (~tmp_sum_pred.isnan()).squeeze()
        non_inf_pred_inds = (~tmp_sum_pred.isinf()).squeeze()
        non_nan_target_inds = (~tmp_sum_target.isnan()).squeeze()
        non_inf_target_inds = (~tmp_sum_target.isinf()).squeeze()

        pred_inds = non_nan_pred_inds & non_inf_pred_inds
        target_inds = non_nan_target_inds & non_inf_target_inds

        if isinstance(foreground_pix, torch.Tensor):
            assert pred_shape == target_shape
            foreground_pix = foreground_pix.to("cuda")
            pred_inds = pred_inds & foreground_pix.squeeze().flatten()
            target_inds = target_inds & foreground_pix.squeeze().flatten()

        elif isinstance(foreground_pix, (list, tuple)):
            if isinstance(foreground_pix[0], torch.Tensor):
                pred_pix = foreground_pix[0].to("cuda")
                pred_inds = pred_inds & pred_pix.squeeze().flatten()
            if isinstance(foreground_pix[1], torch.Tensor):
                target_pix = foreground_pix[1].to("cuda")
                target_inds = target_inds & target_pix.squeeze().flatten()

        pred_pc = pred_pc[:, pred_inds, ...]
        target_pc = target_pc[:, target_inds, ...]

        target_pc = torch.tensor(target_pc, dtype=torch.float32).to("cuda")
        pred_pc = torch.tensor(pred_pc, dtype=torch.float32).to("cuda")
        dist1, dist2, idx1, idx2 = self.chamfer_distance_module(pred_pc, target_pc)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))

        self.sum_chamfer += loss.to("cpu")

        self.num_samples += 1

    def compute(self):
        return self.sum_chamfer / self.num_samples


class BlurredDepthMSE(DepthMSE):
    def __init__(
        self,
        kernel_size=(5, 5),
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.kernel_size = kernel_size

    def update(self, preds, target):
        pred_dist = preds[:, 0, ...]  # Extract only the predicted mean, results in [B, H, W] shape
        pred_dist = pred_dist.unsqueeze(1)  # Gives [B, 1, H, W]
        target = target.unsqueeze(1)  # Gives [B, 1, H, W]

        blurred_pred = tv.transforms.functional.gaussian_blur(pred_dist, self.kernel_size)
        blurred_gt = tv.transforms.functional.gaussian_blur(target, self.kernel_size)

        super().update(blurred_pred, blurred_gt.squeeze())

    def compute(self):
        return super().compute()
