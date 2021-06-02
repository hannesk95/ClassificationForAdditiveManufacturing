import logging
import train
import wandb
import configuration
from torchvision.transforms import transforms
from src.deep_learning.ArchitectureSelector import ArchitectureSelector
from src.deep_learning.DataLoader import VW_Data
from src.deep_learning.ParamConfigurator import ParamConfigurator
from src.deep_learning.NetworkTrainer import NetworkTrainer


def main():

    # 1. Define configuration parameters
    # config = ParamConfigurator()

    params = dict(epochs=configuration.training_configuration.number_epochs,
                  batch_size=configuration.training_configuration.batch_size,
                  lr=configuration.training_configuration.learning_rate,
                  dataset_size=configuration.train_data_configuration.training_data_size,
                  resnet_depth=configuration.training_configuration.resnet_depth)

    # 2. Select neural network architecture and create model
    # selector = ArchitectureSelector(config.architecture_type, config)
    # model = selector.select_architecture()

    # 3. Initialize Dataset
    # data_transforms = transforms.Compose([transforms.ToTensor()])

    # train_set_loader = VW_Data(config.train_data_dir)
    # validation_set_loader = VW_Data(config.validation_data_dir)

    # 4. Start training
    # trainer = NetworkTrainer(model, train_set_loader, validation_set_loader, config)
    # trainer.start_training()

    wandb.login()
    model = train.wandb_initiliazer(params)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info('Started main_deep_learning')

    main()

    logging.info('Finished main_deep_learning')