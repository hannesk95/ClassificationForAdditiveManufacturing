

import sys

sys.path.append(".")   #TODO Ugly - currently needed for LRZ AI System - find better solution
sys.path.append("..")
sys.path.append("../..")


import optuna
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from mlflow import pytorch
from pprint import pformat

import pytorch_lightning as pl
from src.deep_learning.AMCDataset import AMCDataset
from src.deep_learning.ParamConfigurator import ParamConfigurator
from src.deep_learning.ClassificationTask import ClassificationTask
from src.deep_learning.ArchitectureSelector import ArchitectureSelector
from src.deep_learning.PerformanceAnalyst import PerformanceAnalyst


def suggest_hyperparameters(trial):
    # Learning rate on a logarithmic scale
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    # Dropout ratio in the range from 0.0 to 0.9 with step size 0.1
    dropout = trial.suggest_float("dropout", 0.0, 0.9, step=0.1)
    # Optimizer to use as categorical value
    optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "Adadelta"])

    return lr, dropout, optimizer_name


def objective(trial):

    with mlflow.start_run():
        config = ParamConfigurator()
        lr, dropout, optimizer_name = suggest_hyperparameters(trial)
        config.learning_rate = lr
        config.optimizer = optimizer_name
        config.InceptionNet.dropout = dropout
        mlflow.log_params(trial.params)

        # 1. Select neural network architecture and create model
        selector = ArchitectureSelector(config=config)
        nn_model = selector.select_architecture()
        # 3. Define transformations
        transformations = transforms.Compose([transforms.ToTensor()])

        # 4. Initialize dataset
        dataset = AMCDataset(config, transform=transformations)

        # 5. Split dataset into train and val set
        train_data, val_data = random_split(dataset,
                                            [int(config.data_len * config.train_split),
                                             config.data_len - int(config.data_len * config.train_split)],
                                            generator=torch.Generator().manual_seed(42))

        # 6. Create dataloader
        train_data_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, **config.kwargs)
        validation_data_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, **config.kwargs)

        mlflow.log_params(trial.params)

        # 7 . Initialize network
        classifier = ClassificationTask(nn_model=nn_model, config=config)

        logger = DictLogger(trial.number)
        trainer = pl.Trainer(max_epochs=config.num_epochs, deterministic=True, accelerator='horovod', gpus=1,
                             early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="accuracy"),
                             precision=16)
        trainer.fit(classifier)
        mlflow.log_metric("accuracy", logger.metrics[-1]["accuracy"])
    return logger.metrics[-1]["accuracy"]


def main():
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
