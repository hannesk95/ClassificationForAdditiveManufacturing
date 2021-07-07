import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torch.optim import optimizer
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
        self.best_accuracy = 0

    def training_step(self, batch, batch_idx) -> dict:
        """#TODO: Docstring"""
        model, label = batch
        pred = self.nn_model(model)
        self.train_loss = F.binary_cross_entropy_with_logits(pred, label)
        self.train_acc = self.accuracy(pred.round().int(), label.int())

        # train_loss_red = self.metric_average(self.train_loss, 'avg_loss')
        # train_acc_red = self.metric_average(self.train_acc, 'avg_acc')
        #
        # if hvd.rank() == 0:
        #     mlflow.log_metric("train_loss_step", train_loss_red)
        #     mlflow.log_metric("train_acc_step", train_acc_red)

        self.log('train_loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        
        return self.train_loss

    def training_epoch_end(self, training_step_outputs) -> None:
        """#TODO: Docstring"""
        # mlflow.log_metric("train_loss_epoch", self.tensor2float(self.train_loss))
        # mlflow.log_metric("train_acc_epoch", self.tensor2float(self.train_acc))

    def validation_step(self, batch, batch_idx) -> dict:
        """#TODO: Docstring"""
        model, label = batch
        pred = self.nn_model(model)
        self.val_loss = F.binary_cross_entropy_with_logits(pred, label)
        self.val_acc = self.accuracy(pred.round().int(), label.int())

        # val_loss_red = self.metric_average(self.val_loss, 'avg_val_loss')
        # val_acc_red = self.metric_average(self.val_acc, 'avg_val_acc')
        #
        # if hvd.rank() == 0:
        #     mlflow.log_metric("val_loss_step", val_loss_red)
        #     mlflow.log_metric("val_acc_step", val_acc_red)

        self.log('val_loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return self.val_loss

    def validation_epoch_end(self, validation_step_outputs) -> None:
        """#TODO: Docstring"""
        # mlflow.log_metric("val_loss_epoch", self.tensor2float(self.val_loss))
        # mlflow.log_metric("val_acc_epoch", self.tensor2float(self.val_acc))

        # if self.val_acc > self.best_accuracy:
        #     torch.save(self.nn_model.state_dict(), 'model_parameters.pt')
        #     mlflow.log_artifact('model_parameters.pt', artifact_path="best_model_params")
        #     self.best_accuracy = self.val_acc

    def configure_optimizers(self) -> object:
        """#TODO: Docstring"""
        optim = self.config.optimizer
        lr_scheduler = {'scheduler':torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim,mode='min',patience=3),'monitor':self.train_loss}

        return [optim],[lr_scheduler]

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

        # Save dataset path
        mlflow.log_param("dataset", self.config.data_dir)

        # Save num data samples of dataset
        mlflow.log_param("dataset_total_samples", self.config.data_len)

        # Save train/val ratio
        mlflow.log_param("dataset_train_val_ratio", self.config.train_val_ratio)

        # Save model summary
        self.save_model_summary()
        mlflow.log_artifact("model_summary.txt", artifact_path="model_summary")
        os.remove("model_summary.txt")

    def tensor2float(self, tensor) -> float:
        """Convert PyTorch tensor to float"""
        return np.float(tensor.cpu().detach().numpy())

    def metric_average(self, val, name):
        """#TODO: Docstring"""
        tensor = val.detach().clone()
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    def save_model_summary(self):
        """#TODO: Docstring"""
        orig_stdout = sys.stdout
        f = open('model_summary.txt', 'w')
        sys.stdout = f
        if self.config.device.type == 'cuda':
            summary(self.nn_model.cuda(), (1, 128, 128, 128))
        else:
            summary(self.nn_model, (1, 128, 128, 128))
        sys.stdout = orig_stdout
        f.close()

    # def save_model(self):
    #     """#TODO: Docstring"""
    #     PATH = "/workspace/mount_dir/model/model.pt"
    #     torch.save({'epoch': self.epoch_count, 'model_state_dict': self.nn_model.state_dict(),
    #                 'optimizer_state_dict': self.config.optimizer.state_dict()})