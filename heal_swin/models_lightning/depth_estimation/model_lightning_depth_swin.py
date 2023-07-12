from dataclasses import dataclass, field

import pytorch_lightning as pl
from torchmetrics import MetricCollection
import torch.nn as nn

from heal_swin.models_torch import swin_transformer
from heal_swin.utils import depth_utils
from heal_swin.training import loss_depth_regression
from heal_swin.evaluation import custom_metrics
from heal_swin.models_lightning.depth_estimation.depth_common_config import CommonDepthConfig
from heal_swin.data.depth_estimation.data_spec_depth import DepthDataSpec
from heal_swin.training.optimizer import OptimizerConfig, get_lightning_optimizer_dict


@dataclass
class WoodscapeDepthSwinConfig:
    swin_transformer_config: swin_transformer.SwinTransformerConfig = field(
        default_factory=swin_transformer.SwinTransformerConfig
    )
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    common_depth_config: CommonDepthConfig = field(default_factory=CommonDepthConfig)


class WoodscapeDepthSwin(pl.LightningModule):

    CONFIG_CLASS = WoodscapeDepthSwinConfig
    NAME = "depth_swin"

    def __init__(self, config: WoodscapeDepthSwinConfig, data_spec: DepthDataSpec, data_config):
        super().__init__()
        print("")
        print("Creating a SWIN model for depth estimation")
        print("")

        self.config = config
        self.mlflow_params = {}
        self.mlflow_tags = {}

        self.val_metrics_prefix = ""

        self.use_logvar = config.common_depth_config.use_logvar
        self.data_transform = data_config.common_depth.data_transform
        self.mask_background = data_config.common_depth.mask_background
        self.normalize_data = data_config.common_depth.normalize_data

        layer_norms = {"LayerNorm": nn.LayerNorm}
        if isinstance(config.swin_transformer_config.norm_layer, str):
            config.swin_transformer_config.norm_layer = layer_norms[
                config.swin_transformer_config.norm_layer
            ]

        self.model = swin_transformer.SwinTransformerSys(
            config.swin_transformer_config,
            data_spec=data_spec.replace(f_out=2) if self.use_logvar else data_spec,
        )

        if isinstance(config.common_depth_config.train_uncertainty_after, int):
            assert config.common_depth_config.train_uncertainty_after > 0, (
                "Can't switch loss immediately (got switching after epoch "
                f"{config.train_uncertainty_after}), instead set 'train_uncertainty_after=False' "
                "in WoodscapeDepthCommonConfig."
            )

        self.train_uncertainty_after = config.common_depth_config.train_uncertainty_after

        self.depth_data_statistics = data_spec.data_stats

        self.loss = loss_depth_regression.get_depth_loss(config.common_depth_config)

        metric_dict = {
            "mse": custom_metrics.DepthMSE(),
        }

        if self.use_logvar:
            metric_dict.update(
                {
                    "mean_std": custom_metrics.MeanSTD(),
                    "median_std": custom_metrics.MeanSTDMedian(),
                }
            )

        metrics = MetricCollection(metric_dict)
        self.metric_dict = metrics
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

        self.learning_rate = config.optimizer_config.learning_rate

    def forward(self, x):
        outputs = self.model(x.float())
        outputs[:, 0, ...] = depth_utils.unnormalize_and_retransform(
            data=outputs[:, 0, ...],
            normalization=self.normalize_data,
            data_stats=self.depth_data_statistics,
            data_transform=self.data_transform,
        )
        return outputs

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        imgs = batch["s2_imgs"]
        outputs = self(imgs)
        return outputs

    def on_train_epoch_start(self):
        if (
            isinstance(self.train_uncertainty_after, int)
            and self.current_epoch > self.train_uncertainty_after
            and self.use_logvar
        ):
            self.loss = self.depth_uncertainty_loss

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        loss, _ = self.shared_step(batch, self.train_metrics)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs):
        train_metrics_values = self.train_metrics.compute()

        self.log_dict(train_metrics_values, on_epoch=True, on_step=False)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        loss, preds = self.shared_step(batch, self.val_metrics)
        self.log(self.val_metrics_prefix + "val_loss", loss, on_epoch=True, on_step=False)
        preds = preds.unsqueeze(-1)
        return preds

    def validation_epoch_end(self, outputs):
        val_metrics_values = self.val_metrics.compute()

        pref = self.val_metrics_prefix
        val_metric_values = {pref + key: value for key, value in val_metrics_values.items()}

        self.log_dict(val_metric_values, on_epoch=True, on_step=False)
        self.val_metrics.reset()

    def shared_step(self, batch, metrics):
        imgs, masks = batch

        outputs = self(imgs)
        outputs[:, 0, ...] = depth_utils.transform_and_normalize(
            data=outputs[:, 0, ...],
            normalization=self.normalize_data,
            data_stats=self.depth_data_statistics,
            data_transform=self.data_transform,
        )

        loss = self.loss(outputs, masks, mask_background=self.mask_background)

        outputs[:, 0, ...] = depth_utils.unnormalize_and_retransform(
            data=outputs[:, 0, ...],
            normalization=self.normalize_data,
            data_stats=self.depth_data_statistics,
            data_transform=self.data_transform,
        )
        masks = depth_utils.unnormalize_and_retransform(
            data=masks,
            normalization=self.normalize_data,
            data_stats=self.depth_data_statistics,
            data_transform=self.data_transform,
        )

        metrics.update(outputs, masks)

        return loss, outputs

    def configure_optimizers(self):
        return get_lightning_optimizer_dict(self.parameters(), self.config.optimizer_config)
