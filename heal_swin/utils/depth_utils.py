from typing import Dict, Callable, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torchvision as tv
from torchmetrics import Metric
import healpy as hp
from scipy.spatial.transform import Rotation

from heal_swin.evaluation import custom_metrics
from heal_swin.data.depth_estimation import (
    normalize_depth_data,
    project_depth_on_s2,
)


def get_depth_range_indices_from_metrics(
    metric_dict: Dict[str, Metric],
    distance_map: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    inds_dict = {}
    for key in metric_dict.keys():
        if isinstance(metric_dict[key], custom_metrics.DepthRangeMSE):
            distance_range = metric_dict[key].distance_range

            inds = (
                (distance_range[0] <= distance_map)
                & (distance_map < distance_range[1])
                & ~distance_map.isnan()
            )
            inds_dict[key] = inds

    return inds_dict


def create_ranged_mse_mask(
    metric_dict: Dict[str, Metric],
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    range_inds = get_depth_range_indices_from_metrics(metric_dict=metric_dict, distance_map=target)
    output = torch.zeros_like(target)
    output[...] = float(
        "nan"
    )  # initialise this as nans, hence any pixels not matching any range get a nan value
    output = torch.as_tensor(output, dtype=torch.float32)
    for key in range_inds.keys():
        metric = metric_dict[key]
        inds = range_inds[key]
        metric.update(preds=prediction, target=target)
        ranged_mse = torch.as_tensor(metric.compute(), dtype=torch.float32).cpu()
        metric.reset()
        output[inds] = ranged_mse

    return output


def inverse_mask(mask: torch.Tensor) -> torch.Tensor:
    inf_idcs = mask.isinf()
    mask[mask < 1e-3] = 0

    zero_idcs = mask == 0

    rest_idcs = ~(inf_idcs | zero_idcs)

    mask[inf_idcs] = 0
    mask[zero_idcs] = float("inf")
    mask[rest_idcs] = 1 / mask[rest_idcs]

    return mask


def log_mask(mask: torch.Tensor) -> torch.Tensor:
    device = mask.get_device()
    if device == -1:
        device = "cpu"
    tmp = torch.zeros_like(mask, device=device)
    output = torch.log(mask + tmp)
    return output


def exp_mask(mask: torch.Tensor) -> torch.Tensor:
    device = mask.get_device()
    if device == -1:
        device = "cpu"
    tmp = torch.zeros_like(mask, device=device)
    output = torch.exp(mask + tmp)
    return output


def id(mask: torch.Tensor) -> torch.Tensor:
    return mask


def mask_transform_fcn(transform_type: str) -> Callable[[torch.Tensor], torch.Tensor]:
    transform_fcns = {
        "log": log_mask,
        "inv": inverse_mask,
        "None": id,
        None: id,
    }
    return transform_fcns[transform_type]


def reverse_mask_transform_fcn(transform_type: str) -> Callable[[torch.Tensor], torch.Tensor]:
    reverse_transform_fcns = {
        "log": exp_mask,
        "inv": inverse_mask,
        "None": id,
        None: id,
    }
    return reverse_transform_fcns[transform_type]


def save_histogram(
    data: torch.Tensor,
    plot_title: str,
    file_path: str,
    xlabel: str = None,
    ylabel: str = None,
    num_bins: int = 1000,
) -> None:
    cpu_data = data.cpu().flatten()
    cpu_finite_data = cpu_data[cpu_data < float("inf")]
    cpu_finite_data = cpu_finite_data[~cpu_finite_data.isnan()]
    bin_counts, bin_edges = np.histogram(cpu_finite_data.numpy(), bins=num_bins)
    bin_width = np.diff(bin_edges)[0]
    plt.bar(bin_edges[:-1], np.log(bin_counts), width=bin_width)
    plt.title(plot_title)
    if isinstance(xlabel, str):
        plt.xlabel(xlabel)
    if isinstance(ylabel, str):
        plt.ylabel(ylabel)
    plt.savefig(file_path)
    plt.close()


def transform_and_normalize(
    data: torch.Tensor,
    normalization: str,
    data_stats: normalize_depth_data.DataStats,
    data_transform: str,
) -> torch.Tensor:
    data = mask_transform_fcn(data_transform)(data)

    data = normalize_depth_data.normalize_data(
        data=data,
        data_stats=data_stats,
        norm_type=normalization,
    )
    return data


def unnormalize_and_retransform(
    data: torch.Tensor,
    normalization: str,
    data_stats: normalize_depth_data.DataStats,
    data_transform: str,
) -> torch.Tensor:
    data = normalize_depth_data.unnormalize_data(
        data=data,
        data_stats=data_stats,
        norm_type=normalization,
    )

    data = reverse_mask_transform_fcn(data_transform)(data)

    return data


def id_transformation(
    data: torch.Tensor,
    normalization: str,
    data_stats: normalize_depth_data.DataStats,
    data_transform: str,
) -> torch.Tensor:
    data = transform_and_normalize(
        data=data,
        data_stats=data_stats,
        norm_type=normalization,
        data_transform=data_transform,
    )

    data = unnormalize_and_retransform(
        data=data,
        data_stats=data_stats,
        norm_type=normalization,
        data_transform=data_transform,
    )

    return data


def save_depth_pred_comparison_image(
    pred_dist: torch.Tensor,
    original_ground_truth: torch.Tensor,
    ground_truth_after_transforms: torch.Tensor,
    img: Union[torch.Tensor, np.array, np.ndarray],
    filepath: str,
    metric_dict: Dict[str, Metric],
    std: torch.Tensor = None,
    ranged_mse_map: torch.Tensor = None,
    interpolation_mode: str = "nearest",
) -> None:
    assert (
        torch.count_nonzero(ground_truth_after_transforms.isinf()) == 0
    ), "Got infinite values in the retransformed ground truth"
    assert (
        img.ndim == 3
    ), f"The image does not have 3 axes, need three axes for colour plot, got {img.ndim}"
    assert (
        pred_dist.ndim == 2
    ), f"The predicted distance need to have only two axes, got {pred_dist.ndim}"
    assert (
        original_ground_truth.ndim == 2
    ), f"The original ground truth need to have only two axes, got {original_ground_truth.ndim}"
    assert ground_truth_after_transforms.ndim == 2, (
        "The retransformed ground truth need to have only two axes, "
        f"got {ground_truth_after_transforms.ndim}"
    )
    assert original_ground_truth.shape == ground_truth_after_transforms.shape, (
        "The original ground truth and the retransformed ground truth need to have the same shape, "
        f"got {original_ground_truth.shape} and {ground_truth_after_transforms.shape}"
    )
    assert pred_dist.shape == ground_truth_after_transforms.shape, (
        "The predicted distances and the retransformed ground truth need to have the same shape, "
        f"got {pred_dist.shape} and {ground_truth_after_transforms.shape}"
    )
    if std is not None:
        assert std.shape == ground_truth_after_transforms.shape

    assert isinstance(
        img, (torch.Tensor, np.array, np.ndarray)
    ), f"img must be either torch.tensor, np.array, or np.ndarray, got {type(img)}"
    if img.dtype in [torch.float32, torch.float64, np.float32, np.float64]:
        if not ((0 <= img) & (img <= 1)).all():
            img = img / 255
            img[img < 0] = 0
    elif img.dtype in [torch.uint8, np.uint8]:
        assert ((0 <= img) & (img <= 255)).all()
    else:
        raise RuntimeError(f"Expected img to be float, double, or uint got {img.dtype}")

    interpolation_modes = {
        "nearest": tv.transforms.InterpolationMode.NEAREST,
        "bilinear": tv.transforms.InterpolationMode.BILINEAR,
    }

    mse = custom_metrics.DepthMSE()
    og_gt_retr_gt = mse(
        original_ground_truth.unsqueeze(0).unsqueeze(0), ground_truth_after_transforms
    )

    diff = pred_dist - ground_truth_after_transforms
    rel_indices = ~diff.isnan()

    # --------------- construct ranged mse map --------------------
    if ranged_mse_map is None:
        ranged_mse_map_on_rescaled = True
        ranged_mse_map = create_ranged_mse_mask(
            metric_dict, pred_dist.unsqueeze(0).unsqueeze(0), ground_truth_after_transforms
        )
    else:
        ranged_mse_map_on_rescaled = False
        print_size = pred_dist.shape
        transform = tv.transforms.Resize(
            print_size, interpolation=interpolation_modes[interpolation_mode]
        )
        ranged_mse_map = transform(ranged_mse_map.unsqueeze(0)).squeeze()
    # -------------------------------------------------------------

    vcenter = 0.0

    quotient_gts = torch.log(ground_truth_after_transforms / original_ground_truth)
    quotient_inds = ~quotient_gts.isinf() & ~quotient_gts.isnan()
    if quotient_gts[quotient_inds].numel() != 0:
        vmin_quotient = min(torch.min(quotient_gts[quotient_inds]), -0.00001)
        vmax_quotient = max(torch.max(quotient_gts[quotient_inds]), 0.00001)
    else:
        vmin_quotient = -0.00001
        vmax_quotient = 0.00001
    quotient_divnorm = colors.TwoSlopeNorm(vmin=vmin_quotient, vcenter=0.0, vmax=vmax_quotient)

    mask_cmap = plt.get_cmap("Greys_r")
    diff_cmap = plt.get_cmap("seismic")
    log_mask_cmap = plt.get_cmap("Greys_r")
    diff_zoomed_cmap = plt.get_cmap("seismic")
    vmin = min(torch.min(diff[rel_indices]), -0.01)
    vmax = max(torch.max(diff[rel_indices]), 0.01)

    log_vmin = min(
        torch.min(torch.log(ground_truth_after_transforms[rel_indices])),
        torch.min(torch.log(pred_dist[rel_indices])),
    )
    log_vmax = max(
        torch.max(torch.log(ground_truth_after_transforms[rel_indices])),
        torch.max(torch.log(pred_dist[rel_indices])),
    )

    diff_divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    zoomed_in_range = [
        -0.5,
        0.5,
    ]  # I.e. max half a metre of in either direction
    diff_zoomed_divnorm = colors.TwoSlopeNorm(
        vmin=zoomed_in_range[0], vcenter=vcenter, vmax=zoomed_in_range[1]
    )

    log_zoom_range = np.log(np.abs(zoomed_in_range))
    log_zoom_range[0] = torch.min(torch.log(torch.abs(diff[rel_indices])))

    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(30, 20))

    img = ax[0, 0].imshow(np.transpose(img.numpy(), (1, 2, 0)))
    img_ranged_mse = ax[0, 1].imshow(ranged_mse_map)
    img_log_ranged_mse = ax[0, 2].imshow(torch.log(ranged_mse_map))
    diff_gt_mask_transf_mask = ax[0, 3].imshow(quotient_gts, norm=quotient_divnorm, cmap=diff_cmap)

    img_gt = ax[1, 0].imshow(ground_truth_after_transforms, cmap=mask_cmap)
    img_pred = ax[1, 1].imshow(pred_dist, cmap=mask_cmap)
    img_diff = ax[1, 2].imshow(diff, norm=diff_divnorm, cmap=diff_cmap)
    img_zoomed_diff = ax[1, 3].imshow(diff[:, :], norm=diff_zoomed_divnorm, cmap=diff_zoomed_cmap)

    log_img_gt = ax[2, 0].imshow(
        torch.log(ground_truth_after_transforms),
        vmin=log_vmin,
        vmax=log_vmax,
        cmap=mask_cmap,
    )
    log_img_pred = ax[2, 1].imshow(
        torch.log(pred_dist), vmin=log_vmin, vmax=log_vmax, cmap=mask_cmap
    )
    log_img_diff = ax[2, 2].imshow(torch.log(torch.abs(diff)), cmap=mask_cmap)
    log_img_zoomed_diff = ax[2, 3].imshow(
        torch.log(torch.abs(diff)),
        vmin=log_zoom_range[0],
        vmax=log_zoom_range[1],
        cmap=log_mask_cmap,
    )

    ax[0, 0].set_title("img")
    ax[0, 1].set_title(f"ranged mse, computed on rescaled {ranged_mse_map_on_rescaled}")
    ax[0, 2].set_title(f"log(ranged mse), computed on rescaled {ranged_mse_map_on_rescaled}")
    ax[0, 3].set_title(f"log(retransf gt/og gt), mse={og_gt_retr_gt:.4f}")

    ax[1, 0].set_title("gt")
    ax[1, 1].set_title("pred")
    ax[1, 2].set_title("diff")
    ax[1, 3].set_title("zoomed diff")

    ax[2, 0].set_title("log gt")
    ax[2, 1].set_title("log pred")
    ax[2, 2].set_title("log abs diff")
    ax[2, 3].set_title("zoomed log abs diff")

    if std is not None:
        std_pred = ax[3, 0].imshow(std, cmap=mask_cmap)
        plt.colorbar(std_pred, ax=ax[3, 0])
        ax[3, 0].set_title("std")

    ax[0, 0].set_axis_off()
    ax[0, 1].set_axis_off()
    ax[0, 2].set_axis_off()
    ax[0, 3].set_axis_off()

    ax[1, 0].set_axis_off()
    ax[1, 1].set_axis_off()
    ax[1, 2].set_axis_off()
    ax[1, 3].set_axis_off()

    ax[2, 0].set_axis_off()
    ax[2, 1].set_axis_off()
    ax[2, 2].set_axis_off()
    ax[2, 3].set_axis_off()

    ax[3, 0].set_axis_off()
    ax[3, 1].set_axis_off()
    ax[3, 2].set_axis_off()
    ax[3, 3].set_axis_off()

    plt.colorbar(img_ranged_mse, ax=ax[0, 1])
    plt.colorbar(img_log_ranged_mse, ax=ax[0, 2])
    plt.colorbar(diff_gt_mask_transf_mask, ax=ax[0, 3])
    plt.colorbar(img_gt, ax=ax[1, 0])
    plt.colorbar(img_pred, ax=ax[1, 1])
    plt.colorbar(img_diff, ax=ax[1, 2])
    plt.colorbar(img_zoomed_diff, ax=ax[1, 3])
    plt.colorbar(log_img_gt, ax=ax[2, 0])
    plt.colorbar(log_img_pred, ax=ax[2, 1])
    plt.colorbar(log_img_diff, ax=ax[2, 2])
    plt.colorbar(log_img_zoomed_diff, ax=ax[2, 3])

    plt.savefig(filepath)


def get_ray_angles(
    data: torch.Tensor,
    cal_info: Dict,
    nside: int = 8,
    hp_data: bool = False,
    base_pix: int = 8,
    rotate_pole: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not hp_data:
        height, width = data.shape[-2], data.shape[-1]
        u, v = project_depth_on_s2.get_uv_from_hw(
            height, width, output_resolution=data.shape[-2:]
        )  # generate uv inds from image resolution
        # above line gets uv matching the size of the input
        theta, phi = project_depth_on_s2.project_depth_img_points_to_s2(
            u, v, cal_info, rotate_pole, used_size=data.shape[-2:]
        )

        # theta in [0,pi] from np
        # phi in [0,2pi]
        theta = torch.tensor(theta)
        phi = torch.tensor(phi)
    else:
        npix = hp.pixelfunc.nside2npix(nside)
        ipix = np.arange(npix)
        theta, phi = hp.pixelfunc.pix2ang(nside, ipix, nest=True)
        half_idcs = np.arange(npix * base_pix // 12)
        theta = theta[half_idcs]
        phi = phi[half_idcs]

    return theta, phi


def get_unit_vectors(
    data: torch.Tensor,
    cal_info: Dict,
    nside: int = 8,
    hp_data: bool = False,
    base_pix: int = 8,
    rotate_pole: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hp_data:
        theta, phi = get_ray_angles(
            data,
            cal_info,
            nside=nside,
            hp_data=hp_data,
            rotate_pole=rotate_pole,
        )

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

    else:
        npix = hp.pixelfunc.nside2npix(nside)
        ipix = np.arange(npix)
        x, y, z = hp.pixelfunc.pix2vec(nside, ipix, nest=True)  # get unit directional vectors
        half_idcs = np.arange(npix * base_pix // 12)
        x = x[half_idcs]
        y = y[half_idcs]
        z = z[half_idcs]

    return x, y, z


def create_point_cloud_from_depth_mask(
    data: torch.Tensor,
    cal_info: Dict,
    nside: int = 256,
    hp_data: bool = False,
    base_pix: int = 8,
    rotate_pole: bool = False,
    from_angles: bool = True,
    background_val: float = float("nan"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """data is either [N, H, W], or [N, ind]"""
    device = data.get_device()

    pc_foreground = get_foreground_mask(data, background_val=background_val)

    if device == -1:
        device = "cpu"

    if from_angles:
        theta, phi = get_ray_angles(
            data,
            cal_info,
            nside=nside,
            hp_data=hp_data,
            rotate_pole=rotate_pole,
            base_pix=base_pix,
        )
        theta = torch.tensor(theta)
        phi = torch.tensor(phi)

        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
    else:
        x, y, z = get_unit_vectors(
            data,
            cal_info,
            nside=nside,
            hp_data=hp_data,
            rotate_pole=rotate_pole,
            base_pix=base_pix,
        )

    if not hp_data:
        num_samples = data.shape[0]
        x = x.repeat(num_samples, 1, 1)
        y = y.repeat(num_samples, 1, 1)
        z = z.repeat(num_samples, 1, 1)

    x = torch.tensor(x).to(device) * data
    y = torch.tensor(y).to(device) * data
    z = torch.tensor(z).to(device) * data

    stacked_data = torch.stack([x, y, z])

    if not hp_data:
        return_data = stacked_data.permute((1, 2, 3, 0))  # return shape [1,h,w,3]

        point_cloud = return_data.flatten(start_dim=1, end_dim=2)
        pc_foreground = pc_foreground.flatten(start_dim=1, end_dim=2)

    else:
        point_cloud = stacked_data.permute((1, 2, 0))  # return shape [1,ind,3]

    #  ---------- Rotate point cloud according to extrinsic angles ------------
    quaternion = cal_info["extrinsic"]["quaternion"]
    rot_mat = torch.tensor(Rotation.from_quat(quaternion).as_matrix()).to(device)

    point_cloud = point_cloud.permute(2, 1, 0).squeeze()  # shape (3, N)
    point_cloud = torch.matmul(rot_mat, point_cloud)
    point_cloud = point_cloud[..., None]  # shape (3, N, 1)
    point_cloud = point_cloud.permute(2, 1, 0)  # shape (1, N, 3)

    # Return full point cloud and foreground mask
    return point_cloud, pc_foreground


def mask_flat_with_hp_cutout(
    flat_data: torch.Tensor,
    cal_info: Dict,
    base_pix: int = 8,
    nside: int = 256,
    rotate_pole: bool = False,
    masking_val: float = float("nan"),
) -> torch.Tensor:
    working_data = flat_data.clone()

    input_shape = working_data.shape

    if len(input_shape) == 2:
        working_data = np.expand_dims(working_data, axis=0)

    npix = hp.pixelfunc.nside2npix(nside)  # Total pixels in the hp grid
    ipix = np.arange(npix)  # Get the pixl index array
    theta, phi = hp.pixelfunc.pix2ang(
        nside, ipix, nest=True
    )  # Get the corresponding angles for the hp pixels

    half_idcs = np.arange(npix * base_pix // 12)
    theta = theta[half_idcs]
    phi = phi[half_idcs]

    u, v = project_depth_on_s2.project_depth_s2_points_to_img(
        theta, phi, cal_info, rotate_pole
    )  # Get the ordinary image coordinates

    working_mask = np.ones_like(working_data)
    hp_working_mask = project_depth_on_s2.sample_bilinear(working_mask, v, u).astype(np.float32)

    half_idcs = np.arange(npix * base_pix // 12)
    hp_working_mask = hp_working_mask[:, half_idcs]

    hp_working_mask = hp_working_mask.squeeze()

    hp_working_mask_back = project_depth_on_s2.project_depth_hp_mask_back(
        hp_mask=hp_working_mask,
        cal_info=cal_info,
        output_resolution=1,
        rotate_pole=rotate_pole,
        nside=nside,
        base_pix=base_pix,
        s2_bkgd_class=-1,
    )

    mask = np.expand_dims((hp_working_mask_back == -1), axis=0)

    if mask.shape[-2:] != working_data.shape[-2:]:
        mask = tv.transforms.Resize(
            working_data.shape[-2:], interpolation=tv.transforms.InterpolationMode.NEAREST
        )(torch.tensor(mask)).numpy()

    mask = torch.tensor(mask)
    mask = mask.squeeze()
    working_data = torch.tensor(working_data)
    working_data = working_data.squeeze()
    working_data[mask] = masking_val

    working_data = working_data.view(input_shape)

    assert flat_data.shape == working_data.shape

    return working_data


def get_foreground_mask(data: torch.Tensor, background_val: float = float("nan")) -> torch.Tensor:
    """Returns a boolean foreground mask of data based on specified background value(s)"""

    if isinstance(background_val, (tuple, list)):
        collected_masks = []
        for bval in background_val:
            mask = get_foreground_mask(data, background_val=bval)
            collected_masks.append(mask)

        collected_masks = torch.stack(collected_masks, dim=0)
        return collected_masks.all(dim=0)

    if np.isnan(background_val):
        return ~data.isnan()
    elif np.isinf(background_val):
        return ~data.isinf()
    else:
        return data != background_val
