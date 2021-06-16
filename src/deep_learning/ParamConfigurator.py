import configparser
import torch
import os


class ParamConfigurator:
    """Parameter configurator class for deep learning pipeline."""

    def __init__(self):
        """# TODO: Docstring"""
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Architecture
        self.architecture_type = config['architecture']['architecture_type']
        self.experiment_name = self.architecture_type

        # Training
        self.batch_size = config['training'].getint('batch_size')
        self.num_epochs = config['training'].getint('num_epochs')
        self.learning_rate = config['training'].getfloat('learning_rate')
        self.momentum = config['training'].getfloat('momentum')
        self.plot_frequency = config['training'].getint('plot_frequency')
        self.num_workers = config['training'].getint('num_workers')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            self.kwargs = {'num_workers': self.num_workers, 'pin_memory': True}
        else:
            self.kwargs = {'num_workers': self.num_workers}

        self.optimizer = config['training']['optimizer']
        if self.optimizer not in ['Adam', 'SGD']:
            raise ValueError(f"[ERROR] Chosen optimizer is not valid! Please choose out of ['Adam, 'SGD].")

        self.loss_function = config['training']['loss_function']
        if self.loss_function == 'BCE':
            self.loss_function = torch.nn.BCELoss()

        # Dataset
        self.data_dir = config['dataset']['data_dir']
        self.train_val_ratio = config['dataset']['train_val_ratio']
        self.train_split = int(self.train_val_ratio.split("/")[0])/100
        self.val_split = int(self.train_val_ratio.split("/")[-1])/100
        self.data_len = None

        # Train Data
        self.train_data_dir = config['train_data']['train_data_dir']
        if not os.path.exists(self.train_data_dir):
            raise ValueError(f"[ERROR] Directory specified for training data does not exist!")

        # Validation Data
        self.validation_data_dir = config['validation_data']['validation_data_dir']
        if not os.path.exists(self.validation_data_dir):
            raise ValueError(f"[ERROR] Directory specified for validation data does not exist!")

        # ResNet
        self.resnet_depth = config['ResNet'].getint('depth')
        if self.resnet_depth not in [18, 50, 101, 152]:
            raise ValueError(f"[ERROR] ResNet is only available for depths of [18, 50, 101, 152].")
        self.resnet_pretrained = config['ResNet'].getboolean('pretrained')
        if self.resnet_pretrained and self.resnet_depth != 18:
            raise ValueError(f"[ERROR] Only ResNet18 is available with pretrained weights!")

        # Inception
        self.inception_version = config['InceptionNet'].getint('version')
        if self.inception_version not in [int(1), int(3)]:
            raise ValueError(f"[ERROR] InceptionNet is only available for version 1 and for version 3.")

        # MLflow
        self.mlflow_log_dir = config['MLflow']['log_dir']

