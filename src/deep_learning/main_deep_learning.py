import sys
import torch.utils.data
sys.path.append(".")   #TODO Ugly - currently needed for LRZ AI System - find better solution
sys.path.append("..")
sys.path.append("../..")
import logging
import train
import wandb
import configuration
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from src.deep_learning.ArchitectureSelector import ArchitectureSelector
from src.deep_learning.AMCDataset import AMCDataset
from src.deep_learning.ParamConfigurator import ParamConfigurator
from src.deep_learning.NetworkTrainer import NetworkTrainer


def main():

    # 1. Define configuration parameters
    config = ParamConfigurator()

    # 2. Select neural network architecture and create model
    selector = ArchitectureSelector(config.architecture_type, config)
    model = selector.select_architecture()

    # 3. Define transformations
    transformations = transforms.Compose([transforms.ToTensor()])

    # 3. Initialize dataset
    train_dataset = AMCDataset(config.train_data_dir, transform=transformations, cutoff=10)
    validation_dataset = AMCDataset(config.validation_data_dir, transform=transformations, cutoff=10)

    # 4. Create dataloader
    if config.device.type == 'cuda':

        # Training
        config.train_sampler = torch.utils.data.DistributedSampler(train_dataset,
                                                            num_replicas=config.hvd_size, rank=config.hvd_rank)
        train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False,
                                       sampler=config.train_sampler, **config.kwargs)

        # Validation
        config.validation_sampler = torch.utils.data.DistributedSampler(validation_dataset,
                                                                 num_replicas=config.hvd_size, rank=config.hvd_rank)
        validation_data_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False,
                                            sampler=config.validation_sampler, **config.kwargs)
    else:
        train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, **config.kwargs)
        validation_data_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False,
                                            **config.kwargs)

    # 5. Start training
    trainer = NetworkTrainer(model, train_data_loader, validation_data_loader, config)
    trainer.start_training()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info('Started main_deep_learning')

    main()

    logging.info('Finished main_deep_learning')
