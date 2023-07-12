from typing import Callable
from functools import partial

import torch

from heal_swin.models_lightning.depth_estimation.depth_common_config import CommonDepthConfig


def mse(preds: torch.Tensor, target: torch.Tensor, mask_background: bool = False) -> torch.Tensor:

    means = preds[:, 0, ...]
    means = means.unsqueeze(1)

    target = target.unsqueeze(1)  # Get target to same shape as preds: B1HW
    idxs = ~target.isinf().detach()

    sq_diff = torch.square(means[idxs] - target[idxs]) / 2

    loss = torch.mean(sq_diff)
    return loss


def mean_log_var_loss(
    preds: torch.Tensor, target: torch.Tensor, mask_background: bool = False
) -> torch.Tensor:

    means = preds[:, 0, ...]
    log_var = preds[:, 1, ...]

    # IMPORTANT: this is needed to decouple the indices from the torch autograd
    idxs = ~target.isinf().detach()

    std_weighted_sq_diff = 1 / 2 * log_var[idxs] + torch.square(means[idxs] - target[idxs]) * (
        0.5 * torch.exp(-log_var[idxs])
    )
    loss = torch.mean(std_weighted_sq_diff)

    return loss


def l1_loss(
    preds: torch.Tensor, target: torch.Tensor, mask_background: bool = False
) -> torch.Tensor:
    means = preds[:, 0, ...]
    means = means.unsqueeze(1)

    target = target.unsqueeze(1)  # Get target to same shape as preds: B1HW
    idxs = ~target.isinf().detach()

    l1_dist = torch.abs(means[idxs] - target[idxs])

    loss = torch.mean(l1_dist)
    return loss


def huber_loss(
    preds: torch.Tensor, target: torch.Tensor, mask_background: bool = False, delta=1
) -> torch.Tensor:
    means = preds[:, 0, ...]
    means = means.unsqueeze(1)

    target = target.unsqueeze(1)  # Get target to same shape as preds: B1HW
    idxs = ~target.isinf().detach()

    loss = torch.nn.SmoothL1Loss(reduction="mean", beta=delta)(preds[idxs], target[idxs])

    return loss


def get_depth_loss(
    common_depth_config: CommonDepthConfig,
) -> Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor]:
    if common_depth_config.use_logvar:
        print("Only mse base loss available for uncertainty estimation")
        return mean_log_var_loss

    losses = {
        "l2": mse,
        "l1": l1_loss,
        "huber": partial(huber_loss, delta=common_depth_config.huber_delta),
    }

    return losses[common_depth_config.loss]
