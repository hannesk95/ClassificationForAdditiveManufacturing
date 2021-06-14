import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import mlflow
import numpy as np


class ClassificationTask(pl.LightningModule):

    def __init__(self, nn_model: object, config: object):
        super().__init__()
        self.nn_model = nn_model
        self.config = config
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def training_step(self, batch, batch_idx) -> dict:
        model, label = batch
        pred = self.nn_model(model)
        loss = F.binary_cross_entropy(pred, label)
        self.train_acc(pred.round().int(), label.int())
        # acc = torchmetrics.functional.accuracy(pred.round().int(), label.int())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_epoch_end(self, training_step_outputs) -> None:
        pass
        # mlflow.log_metric("loss", 0.1)

    def validation_step(self, batch, batch_idx) -> dict:
        model, label = batch
        pred = self.nn_model(model)
        val_loss = F.binary_cross_entropy(pred, label)
        self.val_acc(pred.round().int(), label.int())
        # val_acc = torchmetrics.functional.accuracy(pred.round(), label)

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return val_loss

    def validation_epoch_end(self, validation_step_outputs) -> None:
        pass
        # mlflow.log_metric("val_loss", 0.1)
        # mlflow.log_metric(list(validation_step_outputs)[0], validation_step_outputs[list(validation_step_outputs)[0]])
        # mlflow.log_metric(list(validation_step_outputs)[1], validation_step_outputs[list(validation_step_outputs)[1]])

    def configure_optimizers(self) -> object:
        return self.config.optimizer

    def get_progress_bar_dict(self) -> dict:
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict

