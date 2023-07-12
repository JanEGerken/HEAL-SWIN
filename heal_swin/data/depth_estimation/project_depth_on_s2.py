import argparse
import os
import json
from tqdm import tqdm
from datetime import datetime
from functools import lru_cache

import numpy as np
import torch
import torchvision as tv
import healpy as hp
import scipy
from scipy import optimize, interpolate
from PIL import Image, ImageDraw

from heal_swin.data.depth_estimation import flat_depth_datasets
from heal_swin.utils import utils, get_paths, healpy_utils

# Image coordinates:
# origin: left upper corner
# u coordinates: along width direction
# v coordinates: along height direction
# shape of image tensors: (C, H, W), so color, v, u


def sample_within_bounds(signal, x, y, bounds, background_value):
    """from the original S2CNN code"""
    xmin, xmax, ymin, ymax = bounds

    idxs = (
        (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)
    )  # Gets all (x,y) that are inside the bounds

    if len(signal.shape) > 2:
        sample = np.full(
            (signal.shape[0], *x.shape), background_value
        )  # Constructs a full sized "image" with background_value everywhere
        sample[:, idxs] = signal[
            :, x[idxs], y[idxs]
        ]  # Sets all eligable points to the actual value, leaving all the rest to background_value
    else:
        sample = np.full(x.shape, background_value, dtype=np.float32)
        sample[idxs] = signal[x[idxs], y[idxs]]

    return sample


def sample_bilinear(signal, rx, ry):
    """adapted from the original S2CNN code

    note: since we use the first dimension of signal as x and the second as y, x corresponds to v
    and y to u
    """

    signal_dim_x = signal.shape[1]
    signal_dim_y = signal.shape[2]

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
    v_int = np.around(v, 0).astype(int)  # Round to nearest ints
    return sample_within_bounds(mask, u_int, v_int, bounds, s2_bkgd_class).astype(np.float32)


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
    theta_rot = np.arccos(xyz_rot[:, 2])
    phi_rot = phi_rot.reshape(phi.shape)
    theta_rot = theta_rot.reshape(theta.shape)

    return theta_rot, phi_rot


@lru_cache(maxsize=23)
def project_depth_s2_points_to_img_cached(
    theta,
    phi,
    theta_shape,
    aspect_ratio,
    cx_offset,
    cy_offset,
    width,
    height,
    poly_order,
    ks,
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


def project_depth_s2_points_to_img(theta, phi, cal_info, rotate_pole, used_size=None):
    """Returns pixel coordinates (floats) corresponding to spherical points"""

    if rotate_pole:
        theta, phi = rot_grid(theta, phi, cal_info, inv=False)

    # Make hashable for caching:
    aspect_ratio = cal_info["intrinsic"]["aspect_ratio"]
    cx_offset = cal_info["intrinsic"]["cx_offset"]
    cy_offset = cal_info["intrinsic"]["cy_offset"]
    if used_size is None:
        width = int(cal_info["intrinsic"]["width"])  # Default (original) size
        height = int(cal_info["intrinsic"]["height"])
    else:
        height = int(used_size[0])
        width = int(used_size[1])
    poly_order = cal_info["intrinsic"]["poly_order"]
    ks = tuple([cal_info["intrinsic"]["k" + str(order)] for order in range(1, poly_order + 1)])

    u, v = project_depth_s2_points_to_img_cached(
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
def project_depth_img_points_to_s2_cached(
    u,
    v,
    u_shape,
    aspect_ratio,
    cx_offset,
    cy_offset,
    width,
    height,
    ks,
    def_width,
    def_height,
):
    """Projects points in the image plane to points on S^2"""

    # Convert hashable inputs back into arrays
    u = np.fromstring(u).reshape(u_shape)
    v = np.fromstring(v).reshape(u_shape)
    u = u * def_width / width
    v = v * def_height / height

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


def project_depth_img_points_to_s2(u, v, cal_info, rotate_pole, used_size=None):

    # Make hashable for caching:
    aspect_ratio = cal_info["intrinsic"]["aspect_ratio"]
    cx_offset = cal_info["intrinsic"]["cx_offset"]
    cy_offset = cal_info["intrinsic"]["cy_offset"]
    def_width = int(cal_info["intrinsic"]["width"])  # Default (original) size
    def_height = int(cal_info["intrinsic"]["height"])
    if used_size is None:
        width = int(cal_info["intrinsic"]["width"])  # Default (original) size
        height = int(cal_info["intrinsic"]["height"])
    else:
        height = int(used_size[0])
        width = int(used_size[1])
    poly_order = cal_info["intrinsic"]["poly_order"]
    ks = tuple([cal_info["intrinsic"]["k" + str(order)] for order in range(1, poly_order + 1)])

    theta, phi = project_depth_img_points_to_s2_cached(
        u=u.tostring(),
        v=v.tostring(),
        u_shape=u.shape,
        aspect_ratio=aspect_ratio,
        cx_offset=cx_offset,
        cy_offset=cy_offset,
        width=width,
        height=height,
        def_width=def_width,
        def_height=def_height,
        ks=ks,
    )

    if rotate_pole:
        theta, phi = rot_grid(theta, phi, cal_info, inv=True)

    return theta, phi


def plot_s2_img(theta, phi, color_vals, title, save_path=None):
    import plotly.graph_objects as go

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    x = x.flatten()
    y = y.flatten()
    z = -z.flatten()

    r_grid = scipy.spatial.transform.Rotation.from_euler("x", [np.pi / 2.0])
    xyz_rot = r_grid.apply(np.stack((x, y, z), axis=-1))
    x = xyz_rot[:, 0]
    y = xyz_rot[:, 1]
    z = xyz_rot[:, 2]
    color_vals = np.einsum("x...->...x", color_vals).reshape(
        -1, 3
    )  # color_vals.transpose(1, 2, 0).reshape(-1, 3)
    color_strs = [f"rgb({c[0]},{c[1]},{c[2]})" for c in color_vals]
    marker_data = go.Scatter3d(
        x=x,
        y=y,
        z=-z,
        marker=go.scatter3d.Marker(size=0.25, color=color_strs),
        mode="markers",
    )
    fig = go.Figure(data=marker_data)
    fig.update_layout(title=title)
    if save_path is None:
        fig.show()
    else:
        fig.write_image(save_path)


def save_metadata(args, save_dir, theta, phi):
    metadata = {
        "bandwidth": args.bandwidth,
        "grid_type": "HealPix",
        "created": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        "samples": args.samples,
        "part": args.part,
        "cam_pos": args.cam_pos,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    np.savez(os.path.join(save_dir, "grid.npz"), theta, phi)


def get_uv_from_cal(cal_info, output_resolution):
    """get pixel coordinates as meshgrid from calibration information
    and desired output resolution"""

    width = cal_info["intrinsic"]["width"]
    height = cal_info["intrinsic"]["height"]

    u_range = np.linspace(0, width - 1, int(width * output_resolution))
    v_range = np.linspace(0, height - 1, int(height * output_resolution))
    u, v = np.meshgrid(u_range, v_range, indexing="xy")
    return u, v


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
    u, v = get_uv_from_cal(cal_info, output_resolution)

    theta, phi = project_depth_img_points_to_s2(u, v, cal_info, rotate_pole)

    u_new, v_new = project_depth_s2_points_to_img(theta, phi, cal_info, rotate_pole)
    print("Mean square projection error in theta:", ((u_new - u) ** 2).mean())
    print("Mean square projection error in phi:", ((v_new - v) ** 2).mean())


def project_depth_hp_img_back(hp_img, cal_info, output_resolution, rotate_pole, base_pix):
    """project image on S^2 back onto flat image plane"""

    u, v = get_uv_from_cal(cal_info, output_resolution)

    theta, phi = project_depth_img_points_to_s2(u, v, cal_info, rotate_pole)

    hp_img_full = np.full((hp_img.shape[0], hp_img.shape[1] * 12 // base_pix), 255.0)
    hp_img_full[:, : hp_img.shape[1]] = hp_img
    hp_img = hp_img_full

    img_new = np.array([hp.get_interp_val(hp_img[i], theta, phi, nest=True) for i in range(3)])
    return img_new


def project_depth_hp_mask_back(
    hp_mask, cal_info, output_resolution, rotate_pole, nside, base_pix, s2_bkgd_class
):
    """project mask on S^2 back onto flat image plane"""
    u, v = get_uv_from_cal(cal_info, output_resolution)

    theta, phi = project_depth_img_points_to_s2(
        u, v, cal_info, rotate_pole
    )  # Gets angles from img indices

    hp_mask_full = np.full((hp_mask.shape[0] * 12 // base_pix,), s2_bkgd_class, dtype=np.float32)
    hp_mask_full[: hp_mask.shape[0]] = hp_mask
    hp_mask = hp_mask_full

    mask_new = np.array([hp.get_interp_val(hp_mask, theta, phi, nest=True)])

    return mask_new


def project_depth_dataset_hp(dataset, args):
    img_save_dir = os.path.join(get_paths.get_syn_datasets_path(), args.save_imgs_to)

    os.makedirs(img_save_dir, exist_ok=True)

    max_idx = len(dataset) if args.samples == -1 else args.samples
    npix = hp.pixelfunc.nside2npix(args.nside)  # Total pixels in the hp grid
    ipix = np.arange(npix)  # Get the pixl index array
    theta, phi = hp.pixelfunc.pix2ang(
        args.nside, ipix, nest=True
    )  # Get the corresponding angles for the hp pixels

    half_idcs = np.arange(npix * args.base_pix // 12)
    theta = theta[half_idcs]
    phi = phi[half_idcs]

    save_metadata(args, img_save_dir, theta, phi)

    for idx in tqdm(range(max_idx)):
        img, mask, cal_info, file_name = dataset[idx]
        u, v = project_depth_s2_points_to_img(theta, phi, cal_info, args.rotate_pole)

        hp_img = sample_bilinear(img, v, u).astype(np.float32)
        hp_mask = sample_mask(
            mask, v, u, args.s2_bkgd_class
        )  # This includes the s2_bkgd_class in the mask

        if args.project_depth_back:
            img_new = project_depth_hp_img_back(
                hp_img,
                cal_info,
                args.output_resolution,
                args.rotate_pole,
                args.base_pix,
            )
            mask_new = project_depth_hp_mask_back(
                hp_mask=hp_mask,
                cal_info=cal_info,
                output_resolution=args.output_resolution,
                rotate_pole=args.rotate_pole,
                nside=args.nside,
                base_pix=args.base_pix,
                s2_bkgd_class=args.s2_bkgd_class,
            )
            np.savez(
                os.path.join(img_save_dir, file_name + ".npz"),
                img=img_new,
                mask=mask_new,
            )
        else:
            np.savez(
                os.path.join(img_save_dir, file_name + ".npz"),
                hp_img=hp_img,
                hp_mask=hp_mask,
            )

    if args.plot_last_compare_s2_flat:
        overlay = utils.get_overlay(mask, img)

        if not args.dont_plot_overlay:
            overlay_new = utils.get_overlay(mask_new, img_new, mask_opacity=0.7)
        else:
            overlay_new = torch.as_tensor(img_new)

        if not args.dont_plot_overlay:
            overlay_new = utils.get_overlay(mask_new, img_new)
        else:
            overlay_new = torch.as_tensor(img_new)
        if args.mark_pole:
            theta_pole, phi_pole = np.array([[0]]), np.array([[0]])
            u_pole, v_pole = project_depth_s2_points_to_img(
                theta_pole, phi_pole, cal_info, args.rotate_pole
            )
            u_pole = u_pole[0, 0] * args.output_resolution
            v_pole = v_pole[0, 0] * args.output_resolution

            im = Image.fromarray(np.transpose(overlay_new.numpy(), (1, 2, 0)))
            draw = ImageDraw.Draw(im)
            size = int(10 * args.output_resolution)
            draw.line(
                [(u_pole - size, v_pole - size), (u_pole + size, v_pole + size)],
                width=2,
                fill="red",
            )
            draw.line(
                [(u_pole - size, v_pole + size), (u_pole + size, v_pole - size)],
                width=2,
                fill="red",
            )
            overlay_new = torch.as_tensor(np.transpose(np.array(im), (2, 0, 1)))

        output_size = (overlay_new.shape[-2], overlay_new.shape[-1])
        overlay_transform = tv.transforms.Resize(output_size)
        overlay = overlay_transform(overlay)

        overlays = torch.stack((overlay, overlay_new)).long() / 255

        pole_adjusted = "_pole_adjusted" if args.rotate_pole else ""
        nside_str = f"_nside={args.nside}"
        base_pix = f"_base_pix={args.base_pix}"
        file_name_new = file_name + f"_back_projected{nside_str}{base_pix}{pole_adjusted}.png"
        tv.utils.save_image(overlays, os.path.join(img_save_dir, file_name_new), nrow=2)

    if args.plot_last_on_s2:
        overlay = utils.get_overlay(hp_mask, hp_img)
        pole_adjusted = "_pole_adjusted" if args.rotate_pole else ""
        base_pix = f"_base_pix={args.base_pix}"
        save_name = file_name + f"_on_s2_nside={args.nside}{base_pix}{pole_adjusted}.png"
        save_path = os.path.join(img_save_dir, save_name)
        healpy_utils.plot_hp_img(
            torch.tensor(hp_img), npix, save_path, projection="orthview", n_colors=12
        )


def add_args(parent_parser):
    parent_parser.add_argument(
        "--cam_pos", type=str, choices=["fv", "rv", "mvr", "mvl"], default=None
    )
    parent_parser.add_argument("--part", type=str, choices=["train", "val", "both"], default="both")
    parent_parser.add_argument("--train_share", type=float, default=0.8)
    parent_parser.add_argument("--samples", type=int, default=-1)
    parent_parser.add_argument("--bandwidth", type=int, default=60)
    parent_parser.add_argument("--save_imgs_to", type=str, default="depth_s2_images")
    parent_parser.add_argument("--output_resolution", type=float, default=1)
    parent_parser.add_argument("--plot_last_on_s2", action="store_true")
    parent_parser.add_argument("--plot_last_compare_s2_flat", action="store_true")
    parent_parser.add_argument("--dont_plot_overlay", action="store_true")
    parent_parser.add_argument("--crop_green", action="store_true")
    parent_parser.add_argument("--s2_bkgd_class", type=int, default=0)
    parent_parser.add_argument("--rotate_pole", action="store_true")
    parent_parser.add_argument("--mark_pole", action="store_true")
    parent_parser.add_argument("--project_depth_back", action="store_true")
    parent_parser.add_argument("--nside", type=int, default=1)
    parent_parser.add_argument(
        "--grid",
        type=str,
        choices=["hp"],
        help="projection grid on the sphere: hp=HealPix",
    )
    parent_parser.add_argument("--base_pix", type=int, default=8)
    return parent_parser


def main(args):
    grid_dispatch = dict(hp=project_depth_dataset_hp)

    if args.part == "both":
        print("Projecting train dataset")
        dataset = flat_depth_datasets.WoodscapeDepthImagesCalibrationDataset(
            crop_green=args.crop_green,
            cam_pos=args.cam_pos,
            train_share=args.train_share,
            part="train",
        )
        grid_dispatch[args.grid](dataset, args)

        print("Projecting val dataset")
        dataset = flat_depth_datasets.WoodscapeDepthImagesCalibrationDataset(
            crop_green=args.crop_green,
            cam_pos=args.cam_pos,
            train_share=args.train_share,
            part="val",
        )
        grid_dispatch[args.grid](dataset, args)

    else:
        print(f"Projecting {args.part} dataset")
        dataset = flat_depth_datasets.WoodscapeDepthImagesCalibrationDataset(
            crop_green=args.crop_green,
            cam_pos=args.cam_pos,
            train_share=args.train_share,
            part=args.part,
        )
        grid_dispatch[args.grid](dataset, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
    print("Done.")
