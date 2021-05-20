import logging
import train
import wandb
import configuration

params = dict(epochs=configuration.training_configuration.number_epochs,batch_size=configuration.training_configuration.batch_size,lr=configuration.training_configuration.learning_rate,dataset_size=configuration.train_data_configuration.training_data_size,resnet_depth=configuration.training_configuration.resnet_depth)


def main():
    wandb.login()
    model = train.wandb_initiliazer(params)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info('Started main_deep_learning')

    main()

    logging.info('Finished main_deep_learning')