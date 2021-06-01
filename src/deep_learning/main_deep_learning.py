import logging
import train
import wandb
import configuration
from src.deep_learning.network import Vanilla3DCNN, ResNet, VGGNet, InceptionNet_v1, InceptionNet_v3
from src.deep_learning.dataloader import VW_Data
from src.deep_learning.ParamConfigurator import ParamConfigurator


def main():

    # 1. Define configuration parameters
    # config = ParamConfigurator()

    params = dict(epochs=configuration.training_configuration.number_epochs,
                  batch_size=configuration.training_configuration.batch_size,
                  lr=configuration.training_configuration.learning_rate,
                  dataset_size=configuration.train_data_configuration.training_data_size,
                  resnet_depth=configuration.training_configuration.resnet_depth)

    # 2. Define network architecture
    model = Vanilla3DCNN()
    # model = ResNet()
    # model = VGGNet
    # model = InceptionNet_v1
    # model = InceptionNet_v3

    # 3. Dataloader
    # train_set_loader = VW_Data(configuration.train_data_configuration.training_set_dir)
    # validation_set_loader = VW_Data(configuration.validation_data_configuration.validation_set_dir)

    # 4. Start training
    wandb.login()
    model = train.wandb_initiliazer(params)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info('Started main_deep_learning')

    main()

    logging.info('Finished main_deep_learning')