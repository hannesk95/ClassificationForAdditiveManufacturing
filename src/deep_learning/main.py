import train
import wandb
import configuration

params = dict(epochs=configuration.training_configuration.number_epochs,batch_size=configuration.training_configuration.batch_size,lr=configuration.training_configuration.learning_rate,dataset_size=configuration.train_data_configuration.training_data_size,resnet_depth=configuration.training_configuration.resnet_depth)

if __name__ == '__main__':
    wandb.login()
    model = train.wandb_initiliazer(params)