import sys
import torch.utils.data
sys.path.append(".")   #TODO Ugly - currently needed for LRZ AI System - find better solution
sys.path.append("..")
sys.path.append("../..")
import logging
import train
import wandb
import configuration
import pytorch_lightning as pl
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from src.deep_learning.ArchitectureSelector import ArchitectureSelector
from src.deep_learning.AMCDataset import AMCDataset
from src.deep_learning.ParamConfigurator import ParamConfigurator
from src.deep_learning.NetworkTrainer import NetworkTrainer
import mlflow.pytorch


def main():

    # 0. Define configuration parameters
    config = ParamConfigurator()

    # 1. Start MLflow logging
    mlflow.set_tracking_uri(config.mlflow_log_dir)
    mlflow.set_experiment(config.architecture_type)
    mlflow.pytorch.autolog(log_every_n_epoch=1)

    # 2. Select neural network architecture and create model
    selector = ArchitectureSelector(config.architecture_type, config)
    model = selector.select_architecture()

    # 3. Define transformations
    transformations = transforms.Compose([transforms.ToTensor()])

    # 3. Initialize dataset
    train_dataset = AMCDataset(config.train_data_dir, transform=transformations)
    validation_dataset = AMCDataset(config.validation_data_dir, transform=transformations)

    # 4. Create dataloader
    if config.device.type == 'cuda':

        # Training
        # config.train_sampler = torch.utils.data.DistributedSampler(train_dataset)
        train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, **config.kwargs)

        # Validation
        # config.validation_sampler = torch.utils.data.DistributedSampler(validation_dataset)
        validation_data_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, **config.kwargs)
    else:
        train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, **config.kwargs)
        validation_data_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False,
                                            **config.kwargs)

    # 5. Start training
    # trainer = NetworkTrainer(model, train_data_loader, validation_data_loader, config)
    # trainer.start_training()

    trainer = pl.Trainer(accelerator='horovod', gpus=1)
    trainer.fit(model, train_data_loader, validation_data_loader)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info('Started main_deep_learning')

    main()

    logging.info('Finished main_deep_learning')
