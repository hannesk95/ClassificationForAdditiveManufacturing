import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
import mlflow


class ClassificationTask(pl.LightningModule):

    def __init__(self, nn_model: object, config: object):
        super().__init__()
        self.nn_model = nn_model
        self.config = config

    def training_step(self, batch, batch_idx) -> dict:
        model, label = batch
        pred = self.nn_model(model)
        loss = F.binary_cross_entropy(pred, label)
        acc = torchmetrics.functional.accuracy(pred.round(), label)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'train_loss': loss, 'train_accuracy': acc}

    def training_epoch_end(self, training_step_outputs) -> None:
        mlflow.log_metric(list(training_step_outputs)[0], training_step_outputs[list(training_step_outputs)[0]])
        mlflow.log_metric(list(training_step_outputs)[1], training_step_outputs[list(training_step_outputs)[1]])

    def validation_step(self, batch, batch_idx) -> dict:
        model, label = batch
        pred = self.nn_model(model)
        val_loss = F.binary_cross_entropy(pred, label)
        val_acc = torchmetrics.functional.accuracy(pred.round(), label)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': val_loss, 'val_accuracy': val_acc}

    def validation_epoch_end(self, validation_step_outputs) -> None:
        mlflow.log_metric(list(validation_step_outputs)[0], validation_step_outputs[list(validation_step_outputs)[0]])
        mlflow.log_metric(list(validation_step_outputs)[1], validation_step_outputs[list(validation_step_outputs)[1]])

    def configure_optimizers(self) -> object:
        return self.config.optimizer

    def get_progress_bar_dict(self) -> dict:
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict

