import configparser
import torch
import os


class ParamConfigurator:
    """Parameter configurator class for deep learning pipeline."""

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')

        # architecture
        self.architecture_type = config['architecture']['architecture_type']

        # training
        self.batch_size = config['training'].getint('batch_size')
        self.num_epochs = config['training'].getint('num_epochs')
        self.learning_rate = config['training'].getfloat('learning_rate')
        self.num_workers = config['training'].getint('num_workers')
        self.plot_frequency = config['training'].getint('plot_frequency')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = config['training']['optimizer']
        if self.optimizer is 'Adam':
            self.optimizer = torch.optim.Adam(lr=self.learning_rate) # TODO: include model.parameters()
        self.loss_function = config['training']['loss_function']
        if self.loss_function is 'BCE':
            self.loss_function = torch.nn.BCELoss()

        # train_data
        self.train_data_size = config['train_data'].getint('train_data_size')
        self.train_data_dir = config['train_data']['train_data_dir']
        if not os.path.exists(self.train_data_dir):
            raise ValueError(f"[ERROR] Directory specified for training data does not exist!")

        # validation_data
        self.validation_data_size = config['validation_data'].getint('validation_data_size')
        self.validation_data_dir = config['validation_data']['validation_data_dir']
        if not os.path.exists(self.validation_data_dir):
            raise ValueError(f"[ERROR] Directory specified for validation data does not exist!")

        # ResNet
        self.resnet_depth = config['ResNet'].getint('depth')
        if self.resnet_depth not in [18, 50, 101, 152]:
            raise ValueError(f"[ERROR] ResNet is only available for depths of [18, 50, 101, 152].")

        # Inception
        self.inception_version = config['ResNet'].getint('version')
        if self.inception_version not in [1, 3]:
            raise ValueError(f"[ERROR] InceptionNet is only available for version 1 and for version 3.")







