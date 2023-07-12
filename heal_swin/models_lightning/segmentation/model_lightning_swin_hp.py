from typing import Optional, List
from dataclasses import dataclass, field

import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection, IoU, Accuracy

from heal_swin.models_torch.swin_hp_transformer import SwinHPTransformerConfig, SwinHPTransformerSys
from heal_swin.training.optimizer import OptimizerConfig, get_lightning_optimizer_dict
from heal_swin.data.segmentation.data_spec import DataSpec


@dataclass
class WoodscapeSegmenterSwinHPConfig:
    swin_hp_transformer_config: SwinHPTransformerConfig = field(
        default_factory=SwinHPTransformerConfig
    )
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    class_weights: Optional[List[float]] = None


class WoodscapeSegmenterSwinHP(pl.LightningModule):

    CONFIG_CLASS = WoodscapeSegmenterSwinHPConfig
    NAME = "swin_hp"

    def __init__(self, config, data_spec: DataSpec, data_config):

        super().__init__()
        self.config = config
        self.mlflow_params = {}
        self.mlflow_tags = {}

        self.val_metrics_prefix = ""

        self.model = SwinHPTransformerSys(config.swin_hp_transformer_config, data_spec=data_spec)

        if config.class_weights is None:
            weights = torch.ones(data_spec.f_out, dtype=torch.float)
        else:
            assert len(config.class_weights) == data_spec.f_out
            weights = torch.tensor(config.class_weights, dtype=torch.float)

        self.loss = nn.CrossEntropyLoss(weight=weights)

        metrics = MetricCollection(
            {
                "iou": IoU(num_classes=data_spec.f_out, reduction="none"),
                "acc": Accuracy(),
                "acc_ignored": Accuracy(ignore_index=0),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.class_names = data_spec.class_names

        self.learning_rate = config.optimizer_config.learning_rate

    def forward(self, x):
        return self.model(x.float())

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        imgs = batch["hp_imgs"]
        outputs = self(imgs)
        _, preds = torch.max(outputs, 1)
        return preds

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, self.train_metrics)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs):
        train_metrics_values = self.train_metrics.compute()
        iou = train_metrics_values["train_iou"]
        train_metrics_values["train_iou_global_ignored"] = torch.mean(iou[1:])
        train_metrics_values["train_iou_global"] = torch.mean(iou)
        del train_metrics_values["train_iou"]  # not a scalar and hence cannot be logged

        self.log_dict(train_metrics_values, on_epoch=True, on_step=False)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, preds = self.shared_step(batch, self.val_metrics)
        self.log(self.val_metrics_prefix + "val_loss", loss, on_epoch=True, on_step=False)
        return preds

    def validation_epoch_end(self, outputs):
        val_metrics_values = self.val_metrics.compute()
        iou = val_metrics_values["val_iou"]
        for c in range(len(iou)):
            val_metrics_values[f"val_iou_global_class_{c}_{self.class_names[c]}"] = iou[c]
        val_metrics_values["val_iou_global_ignored"] = torch.mean(iou[1:])
        val_metrics_values["val_iou_global"] = torch.mean(iou)
        del val_metrics_values["val_iou"]  # not a scalar and hence cannot be logged

        pref = self.val_metrics_prefix
        val_metric_values = {pref + key: value for key, value in val_metrics_values.items()}

        self.log_dict(val_metric_values, on_epoch=True, on_step=False)
        self.val_metrics.reset()

    def shared_step(self, batch, metrics):
        imgs, masks = batch
        outputs = self(imgs)
        _, preds = torch.max(outputs, 1)

        loss = self.loss(outputs, masks.long())
        metrics.update(preds, masks)
        return loss, preds

    def configure_optimizers(self):
        return get_lightning_optimizer_dict(self.parameters(), self.config.optimizer_config)
