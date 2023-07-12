import argparse
from tqdm import tqdm
import os
import json
from datetime import datetime
from functools import lru_cache

import numpy as np
import scipy
from scipy import optimize, interpolate
import torch
import healpy as hp

from heal_swin.data.segmentation import flat_datasets
from heal_swin.utils import utils, get_paths, healpy_utils

# Image coordinates:
# origin: left upper corner
# u coordinates: along width direction
# v coordinates: along height direction
# shape of image tensors: (C, H, W), so color, v, u


def sample_within_bounds(signal, x, y, bounds, background_value):
    """from the original S2CNN code"""
    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    if len(signal.shape) > 2:
        # import pdb; pdb.set_trace()
        sample = np.full((signal.shape[0], *x.shape), background_value)
        sample[:, idxs] = signal[:, x[idxs], y[idxs]]
    else:
        sample = np.full(x.shape, background_value)
        sample[idxs] = signal[x[idxs], y[idxs]]
    return sample


def sample_bilinear(signal, rx, ry):
    """adapted from the original S2CNN code

    note: since we use the first dimension of signal as x and the second as y, x corresponds to v
    and y to u
    """

    signal_dim_x = signal.shape[1]
    signal_dim_y = signal.shape[2]

    # discretize sample position
    # ix = rx.astype(int)
    # iy = ry.astype(int)

    # obtain four sample coordinates
    ix0 = np.floor(rx).astype(int)
    iy0 = np.floor(ry).astype(int)
    ix1 = np.ceil(rx).astype(int)
    iy1 = np.ceil(ry).astype(int)

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds, 0)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds, 0)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds, 0)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds, 0)

    # linear interpolation in x-direction
    fx1 = (ix1 - rx) * signal_00 + (rx - ix0) * signal_10
    fx2 = (ix1 - rx) * signal_01 + (rx - ix0) * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry) * fx1 + (ry - iy0) * fx2


def sample_mask(mask, u, v, s2_bkgd_class):
    bounds = (0, mask.shape[0], 0, mask.shape[1])
    u_int = np.around(u, 0).astype(int)
    v_int = np.around(v, 0).astype(int)
    return sample_within_bounds(mask, u_int, v_int, bounds, s2_bkgd_class).astype(np.uint8)


@lru_cache(maxsize=23)
def hp_nearest_pix_idcs(theta, phi, theta_shape, nside):
    """To perform nearest neighbor interpolation on the sphere in healpy we get the (four) closest
    pixel indices and weights per sample pixel for a bilinear interpolation in the healpy grid. Then
    we select the pixel with the highest weight as the overall nearest neighbor.

    """
    # Convert hashable inputs back into arrays
    theta = np.fromstring(theta).reshape(theta_shape)
    phi = np.fromstring(phi).reshape(theta_shape)

    pix_idcs, weights = hp.pixelfunc.get_interp_weights(
        nside=nside, phi=phi, theta=theta, nest=True, lonlat=False
    )

    max_weight_idcs = np.argmax(weights, axis=0)

    i, j = np.meshgrid(
        np.arange(max_weight_idcs.shape[0]), np.arange(max_weight_idcs.shape[1]), indexing="ij"
    )

    nearest_pix_idcs = pix_idcs[max_weight_idcs, i, j]

    return nearest_pix_idcs


def rot_grid(theta, phi, cal_info, inv=False):
    r = scipy.spatial.transform.Rotation.from_quat(cal_info["extrinsic"]["quaternion"])
    if cal_info["name"] == "FV":
        ext_ref = [1, 0, 0]
    elif cal_info["name"] == "RV":
        ext_ref = [-1, 0, 0]
    elif cal_info["name"] == "MVL":
        ext_ref = [0, 1, 0]
    elif cal_info["name"] == "MVR":
        ext_ref = [0, -1, 0]
    int_ref = r.inv().apply(ext_ref)
    phi_ref = np.arctan2(int_ref[1], int_ref[0])
    theta_ref = np.arccos(int_ref[2])
    r_grid = scipy.spatial.transform.Rotation.from_euler("yz", [theta_ref, phi_ref])
    if inv:
        r_grid = r_grid.inv()
    x = (np.cos(phi) * np.sin(theta)).reshape(-1)
    y = (np.sin(phi) * np.sin(theta)).reshape(-1)
    z = (np.cos(theta)).reshape(-1)
    xyz_rot = r_grid.apply(np.stack((x, y, z), axis=-1))
    phi_rot = np.arctan2(xyz_rot[:, 1], xyz_rot[:, 0])
    # arctan2 takes values in [-pi,pi], so move this back to [0,2pi] as in the DH grid
    # phi_rot[phi_rot < 0] = phi_rot[phi_rot < 0] + 2 * np.pi
    theta_rot = np.arccos(xyz_rot[:, 2])
    phi_rot = phi_rot.reshape(phi.shape)
    theta_rot = theta_rot.reshape(theta.shape)

    return theta_rot, phi_rot


@lru_cache(maxsize=23)
def project_s2_points_to_img_cached(
    theta, phi, theta_shape, aspect_ratio, cx_offset, cy_offset, width, height, poly_order, ks
):
    theta = np.fromstring(theta).reshape(theta_shape)
    phi = np.fromstring(phi).reshape(theta_shape)

    rho = 0
    for order in range(1, poly_order + 1):
        rho += ks[order - 1] * theta**order
    u = rho * np.cos(phi)
    v = rho * np.sin(phi)
    u = u + cx_offset + width / 2 - 0.5
    v = v * aspect_ratio + cy_offset + height / 2 - 0.5
    return u, v


def project_s2_points_to_img(theta, phi, cal_info, rotate_pole):
    """Returns pixel coordinates (floats) corresponding to spherical points"""

    if rotate_pole:
        theta, phi = rot_grid(theta, phi, cal_info, inv=False)

    # Make hashable for caching:
    aspect_ratio = cal_info["intrinsic"]["aspect_ratio"]
    cx_offset = cal_info["intrinsic"]["cx_offset"]
    cy_offset = cal_info["intrinsic"]["cy_offset"]
    width = int(cal_info["intrinsic"]["width"])
    height = int(cal_info["intrinsic"]["height"])
    poly_order = cal_info["intrinsic"]["poly_order"]
    ks = tuple([cal_info["intrinsic"]["k" + str(order)] for order in range(1, poly_order + 1)])

    u, v = project_s2_points_to_img_cached(
        theta=theta.tostring(),
        phi=phi.tostring(),
        theta_shape=theta.shape,
        aspect_ratio=aspect_ratio,
        cx_offset=cx_offset,
        cy_offset=cy_offset,
        width=width,
        height=height,
        poly_order=poly_order,
        ks=ks,
    )

    return u, v


@lru_cache(maxsize=23)
def project_img_points_to_s2_cached(
    u, v, u_shape, aspect_ratio, cx_offset, cy_offset, width, height, ks
):
    """Projects points in the image plane to points on S^2"""

    # Convert hashable inputs back into arrays
    u = np.fromstring(u).reshape(u_shape)
    v = np.fromstring(v).reshape(u_shape)

    u = u - cx_offset - width / 2 + 0.5
    v = (v - cy_offset - height / 2 + 0.5) / aspect_ratio

    rho = np.sqrt(u**2 + v**2)
    phi = np.arctan2(v, u)
    phi[phi < 0] = 2 * np.pi + phi[phi < 0]

    def f(rho):
        """Returns function to find zero of to invert rho(theta)"""
        return (
            lambda theta: rho
            - ks[0] * theta
            - ks[1] * theta**2
            - ks[2] * theta**3
            - ks[3] * theta**4
        )

    rho_samples = np.linspace(0, rho.max(), 100)
    theta_samples = [optimize.newton_krylov(f(rho), np.pi / 2).item() for rho in rho_samples]
    theta_of_rho = interpolate.interp1d(rho_samples, theta_samples)
    theta = theta_of_rho(rho)

    return theta, phi


def project_img_points_to_s2(u, v, cal_info, rotate_pole):

    # Make hashable for caching:
    aspect_ratio = cal_info["intrinsic"]["aspect_ratio"]
    cx_offset = cal_info["intrinsic"]["cx_offset"]
    cy_offset = cal_info["intrinsic"]["cy_offset"]
    width = int(cal_info["intrinsic"]["width"])
    height = int(cal_info["intrinsic"]["height"])
    poly_order = cal_info["intrinsic"]["poly_order"]
    ks = tuple([cal_info["intrinsic"]["k" + str(order)] for order in range(1, poly_order + 1)])

    theta, phi = project_img_points_to_s2_cached(
        u=u.tostring(),
        v=v.tostring(),
        u_shape=u.shape,
        aspect_ratio=aspect_ratio,
        cx_offset=cx_offset,
        cy_offset=cy_offset,
        width=width,
        height=height,
        ks=ks,
    )

    if rotate_pole:
        theta, phi = rot_grid(theta, phi, cal_info, inv=True)

    return theta, phi


def save_metadata(args, save_dir, theta, phi):
    metadata = {
        "nside": args.nside,
        "base_pix": args.base_pix,
        "grid_type": "HEALPix",
        "created": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        "samples": args.samples,
        "part": args.part,
        "cam_pos": args.cam_pos,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    np.savez(os.path.join(save_dir, "grid.npz"), theta, phi)


def get_uv_from_hw(height, width, output_resolution):
    """get pixel coordinates as meshgrid from calibration information
    and desired output resolution"""

    if isinstance(output_resolution, float):  # scale by factor
        height_res = int(height * output_resolution)
        width_res = int(width * output_resolution)
    elif isinstance(output_resolution, int):  # scale shorter side to value
        if width <= height:
            width_res = output_resolution
            height_res = int(height * output_resolution) // width_res
        else:
            height_res = output_resolution
            width_res = int(width * output_resolution) // height_res
    elif isinstance(output_resolution, tuple):  # scale to exact size
        height_res = output_resolution[0]
        width_res = output_resolution[1]

    u_range = np.linspace(0, width - 1, width_res)
    v_range = np.linspace(0, height - 1, height_res)
    u, v = np.meshgrid(u_range, v_range, indexing="xy")
    return u, v


def test_projection_accuracy_hp(hp_img, cal_info, output_resolution, rotate_pole, base_pix):
    width = cal_info["intrinsic"]["width"]
    height = cal_info["intrinsic"]["height"]
    u, v = get_uv_from_hw(height, width, output_resolution)

    theta, phi = project_img_points_to_s2(u, v, cal_info, rotate_pole)

    u_new, v_new = project_s2_points_to_img(theta, phi, cal_info, rotate_pole)
    print("Mean square projection error in theta:", ((u_new - u) ** 2).mean())
    print("Mean square projection error in phi:", ((v_new - v) ** 2).mean())


def project_hp_img_back(hp_img, cal_info, output_resolution, rotate_pole, base_pix):
    """project image on S^2 back onto flat image plane"""

    width = cal_info["intrinsic"]["width"]
    height = cal_info["intrinsic"]["height"]
    u, v = get_uv_from_hw(height, width, output_resolution)

    theta, phi = project_img_points_to_s2(u, v, cal_info, rotate_pole)

    hp_img_full = np.full((hp_img.shape[0], hp_img.shape[1] * 12 // base_pix), 255.0)
    hp_img_full[:, : hp_img.shape[1]] = hp_img
    hp_img = hp_img_full

    img_new = np.array([hp.get_interp_val(hp_img[i], theta, phi, nest=True) for i in range(3)])
    return img_new


def project_hp_mask_back(
    hp_mask, cal_info, output_resolution, rotate_pole, nside, base_pix, s2_bkgd_class
):
    """project mask on S^2 back onto flat image plane"""
    width = cal_info["intrinsic"]["width"]
    height = cal_info["intrinsic"]["height"]
    u, v = get_uv_from_hw(height, width, output_resolution)

    theta, phi = project_img_points_to_s2(u, v, cal_info, rotate_pole)

    hp_mask_full = np.full((hp_mask.shape[0] * 12 // base_pix,), s2_bkgd_class)
    hp_mask_full[: hp_mask.shape[0]] = hp_mask
    hp_mask = hp_mask_full

    nearest_pix_idcs = hp_nearest_pix_idcs(
        theta=theta.tostring(),
        phi=phi.tostring(),
        theta_shape=theta.shape,
        nside=nside,
    )
    mask_new = hp_mask[nearest_pix_idcs].astype(np.uint8)

    return mask_new


def project_dataset_hp(dataset, args):
    img_save_dir = os.path.join(
        get_paths.get_datasets_path(args.woodscape_version), args.save_imgs_to
    )
    os.makedirs(img_save_dir, exist_ok=True)

    max_idx = len(dataset) if args.samples == -1 else args.samples
    npix = hp.pixelfunc.nside2npix(args.nside)
    ipix = np.arange(npix)
    theta, phi = hp.pixelfunc.pix2ang(args.nside, ipix, nest=True)

    half_idcs = np.arange(npix * args.base_pix // 12)
    theta = theta[half_idcs]
    phi = phi[half_idcs]

    save_metadata(args, img_save_dir, theta, phi)

    for idx in tqdm(range(max_idx)):
        img, mask, cal_info, file_name = dataset[idx]
        u, v = project_s2_points_to_img(theta, phi, cal_info, args.rotate_pole)

        hp_img = sample_bilinear(img, v, u).astype(np.uint8)
        hp_mask = sample_mask(mask, v, u, args.s2_bkgd_class)

        np.savez(
            os.path.join(img_save_dir, file_name + ".npz"),
            hp_img=hp_img,
            hp_mask=hp_mask,
        )

    if args.plot_last_on_s2:
        overlay = utils.get_overlay(args.woodscape_version, hp_mask, hp_img)
        pole_adjusted = "_pole_adjusted" if args.rotate_pole else ""
        base_pix = f"_base_pix={args.base_pix}"
        save_name = file_name + f"_on_s2_nside={args.nside}{base_pix}{pole_adjusted}.png"
        save_path = os.path.join(img_save_dir, save_name)
        healpy_utils.plot_hp_img(
            torch.tensor(overlay), npix, save_path, projection="orthview", n_colors=12
        )


def add_args(parent_parser):
    parent_parser.add_argument(
        "--cam_pos", type=str, choices=["fv", "rv", "mvr", "mvl"], default=None
    )
    parent_parser.add_argument("--part", type=str, choices=["train", "val", "both"], default="both")
    parent_parser.add_argument("--train_share", type=float, default=0)
    parent_parser.add_argument("--samples", type=int, default=-1)
    parent_parser.add_argument("--save_imgs_to", type=str, default="s2_images")
    parent_parser.add_argument("--output_resolution", type=float, default=1)
    parent_parser.add_argument("--plot_last_on_s2", action="store_true")
    parent_parser.add_argument("--plot_last_compare_s2_flat", action="store_true")
    parent_parser.add_argument("--crop_green", action="store_true")
    parent_parser.add_argument("--s2_bkgd_class", type=int, default=0)
    parent_parser.add_argument("--rotate_pole", action="store_true")
    parent_parser.add_argument("--nside", type=int, default=1)
    parent_parser.add_argument("--base_pix", type=int, default=8)
    parent_parser.add_argument("--woodscape_version", type=str)
    return parent_parser


def main(args):

    if args.part == "both":
        print("Projecting train dataset")
        dataset = flat_datasets.WoodscapeSemanticImagesCalibrationDataset(
            crop_green=args.crop_green,
            cam_pos=args.cam_pos,
            train_share=args.train_share,
            part="train",
            woodscape_version=args.woodscape_version,
        )
        project_dataset_hp(dataset, args)

        print("Projecting val dataset")
        dataset = flat_datasets.WoodscapeSemanticImagesCalibrationDataset(
            crop_green=args.crop_green,
            cam_pos=args.cam_pos,
            train_share=args.train_share,
            part="val",
            woodscape_version=args.woodscape_version,
        )
        project_dataset_hp(dataset, args)
    else:
        print(f"Projecting {args.part} dataset")
        dataset = flat_datasets.WoodscapeSemanticImagesCalibrationDataset(
            crop_green=args.crop_green,
            cam_pos=args.cam_pos,
            train_share=args.train_share,
            part=args.part,
            woodscape_version=args.woodscape_version,
        )
        project_dataset_hp(dataset, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
    print("Done.")
