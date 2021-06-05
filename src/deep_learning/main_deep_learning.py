import sys
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

    params = dict(epochs=configuration.training_configuration.number_epochs,
                  batch_size=configuration.training_configuration.batch_size,
                  lr=configuration.training_configuration.learning_rate,
                  dataset_size=configuration.train_data_configuration.training_data_size,
                  resnet_depth=configuration.training_configuration.resnet_depth)

    # 2. Select neural network architecture and create model
    selector = ArchitectureSelector(config.architecture_type, config)
    model = selector.select_architecture()

    # 3. Define transformations
    transformations = transforms.Compose([transforms.ToTensor()])

    # 3. Initialize dataset
    train_dataset = AMCDataset(config.train_data_dir, transform=transformations)
    validation_dataset = AMCDataset(config.validation_data_dir, transform=transformations)

    # 4. Create dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False,
                                   num_workers=config.num_workers, pin_memory=config.pin_memory)
    validation_data_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False,
                                        num_workers=config.num_workers, pin_memory=config.pin_memory)

    # 5. Start training
    trainer = NetworkTrainer(model, train_data_loader, validation_data_loader, config)
    trainer.start_training()

    # wandb.login()
    # model = train.wandb_initiliazer(params)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info('Started main_deep_learning')

    main()

    logging.info('Finished main_deep_learning')
