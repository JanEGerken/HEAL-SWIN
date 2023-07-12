import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import healpy


def plot_healpy_img(
    healpy_img,
    npix,
    path=None,
    show_plot=False,
    n_colors=0,
    opacity=0.5,
    projection="mollview",
    xsize=800,
    nest=True,
    rot=(0, 0, 0),
    bkgd_color=(255, 255, 255),
    s2_bkgd_color=(200, 200, 200),
):
    """Plots a healpy image

    Args:
    healpy_img: torch or numpy tensor of shape (color, healpy pixel), not necessarily full
                healpy grid
    npix: npix of the full healpy grid, the remaining pixels are filled with white
    path: save path of generated image don't save if None
    show_plot: if True, show plot in X1
    n_colors: if > 0, partition image in n_colors consecutive batches and overlay them with
              different colors
    opacity: opacity of the overlay
    projection: projection function to use
    nest: True if image is given in nested ordering
    rot: rotation angles around x,y,z axis before projection
    bkgd_color: Tuple of three 0...255 ints for color outside of projection region
    s2_bkgd_color: Tuple of three 0...255 ints for color on the sphere without pixels
    """
    if isinstance(healpy_img, torch.Tensor):
        healpy_img = healpy_img.detach().cpu().numpy()
    healpy_img = healpy_img.astype(np.float32)

    proj_fct_dispath = {
        "mollview": healpy.visufunc.mollview,
        "cartview": healpy.visufunc.cartview,
        "orthview": healpy.visufunc.orthview,
    }

    assert healpy_img.shape[0] == 3
    assert len(healpy_img.shape) == 2
    assert healpy.isnpixok(npix)
    assert projection in proj_fct_dispath

    proj_fct = proj_fct_dispath[projection]

    if n_colors > 0:
        colors = cm.rainbow(np.linspace(0, 1, n_colors))
        window_size = healpy_img.shape[1] // n_colors
        for window in range(n_colors):
            low_idx = window * window_size
            high_idx = (window + 1) * window_size
            color = opacity * (colors[window, :-1] * 255.0)[:, None]
            healpy_img[:, low_idx:high_idx] += color

    healpy_img /= healpy_img.max() * (1 / 255)
    healpy_img = healpy_img.astype(np.uint8)

    healpy_img_full = np.full((healpy_img.shape[0], npix), np.array(s2_bkgd_color)[:, None])
    healpy_img_full[:, : healpy_img.shape[1]] = healpy_img
    healpy_img = healpy_img_full / 255

    def plot_channel(i):
        proj_fct(healpy_img[i], nest=nest, xsize=xsize, return_projected_map=True, rot=rot)

    img = np.stack([plot_channel(i) for i in range(3)], axis=-1)
    # if -inf in first channel, all channels are -inf:
    img[(img[:, :, 0] == -np.inf), :] = np.array(bkgd_color)[None, :] / 255
    plt.clf()  # clear whatever proj_fct drew
    plt.imshow(img)
    plt.axis("off")
    # plt.tight_layout(pad=0.0)
    if show_plot:
        plt.show()
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
