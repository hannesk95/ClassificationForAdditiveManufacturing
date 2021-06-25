import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy
import mlflow
from torchsummary import summary
import sys
import os
import horovod.torch as hvd


class ClassificationTask(pl.LightningModule):
    """#TODO: Docstring"""

    def __init__(self, nn_model: object, config: object):
        """#TODO: Docstring"""
        super().__init__()
        self.nn_model = nn_model
        self.config = config
        self.accuracy = Accuracy()
        self.train_acc = None
        self.val_acc = None
        self.train_loss = None
        self.val_loss = None
        self.epoch_count = 0
        self.save_mlflow_params()

    def metric_average(self, val, name):
        tensor = val.detach().clone()
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    def training_step(self, batch, batch_idx) -> dict:
        """#TODO: Docstring"""
        model, label = batch
        pred = self.nn_model(model)
        self.train_loss = F.binary_cross_entropy_with_logits(pred, label)
        self.train_acc = self.accuracy(pred.round().int(), label.int())

        # mlflow.log_metric("train_loss_step", self.tensor2float(self.train_loss))
        # mlflow.log_metric("train_acc_step", self.tensor2float(self.train_acc))

        # Horovod: average metric values across workers.
        train_loss = self.metric_average(self.train_loss, 'avg_loss')
        train_acc= self.metric_average(self.train_acc, 'avg_accuracy')
    
        # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            self.log('train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
            self.log('train_acc', train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return self.train_loss

    def training_epoch_end(self, training_step_outputs) -> None:
        """#TODO: Docstring"""
        # self.epoch_count += 1

        # mlflow.log_metric("train_loss_epoch", self.tensor2float(self.train_loss))
        # mlflow.log_metric("train_acc_epoch", self.tensor2float(self.train_acc))

        # if self.epoch_count % 10:
            # print(f"Saving model every 10 epochs...")

    def validation_step(self, batch, batch_idx) -> dict:
        """#TODO: Docstring"""
        model, label = batch
        pred = self.nn_model(model)
        self.val_loss = F.binary_cross_entropy_with_logits(pred, label)
        self.val_acc = self.accuracy(pred.round().int(), label.int())

        # mlflow.log_metric("val_loss_step", self.tensor2float(self.val_loss))
        # mlflow.log_metric("val_acc_step", self.tensor2float(self.val_acc))

        # Horovod: average metric values across workers.
        val_loss = self.metric_average(self.val_loss, 'avg_loss')
        val_acc = self.metric_average(self.val_acc, 'avg_accuracy')

        # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
            self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return self.val_loss

    def validation_epoch_end(self, validation_step_outputs) -> None:
        """#TODO: Docstring"""
        # mlflow.log_metric("val_loss_epoch", self.tensor2float(self.val_loss))
        # mlflow.log_metric("val_acc_epoch", self.tensor2float(self.val_acc))

    def configure_optimizers(self) -> object:
        """#TODO: Docstring"""
        return self.config.optimizer

    def get_progress_bar_dict(self) -> dict:
        """#TODO: Docstring"""
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict

    def save_mlflow_params(self):
        """#TODO: Docstring"""
        # Save optimizer
        mlflow.log_param("optimizer", self.config.optimizer)

        # Save learning rate
        mlflow.log_param("learning_rate", self.config.learning_rate)

        # Save number of epochs
        mlflow.log_param("epochs", self.config.num_epochs)

        # Save model summary
        orig_stdout = sys.stdout
        f = open('model_summary.txt', 'w')
        sys.stdout = f
        if self.config.device.type == 'cuda':
            summary(self.nn_model.cuda(), (1, 128, 128, 128))
        else:
            summary(self.nn_model, (1, 128, 128, 128))
        sys.stdout = orig_stdout
        f.close()
        mlflow.log_artifact("model_summary.txt", artifact_path="model_summary")
        os.remove("model_summary.txt")

    @staticmethod
    def tensor2float(self, tensor) -> float:
        """Convert PyTorch tensor to float"""
        return np.float(tensor.detach().numpy())





