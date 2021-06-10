import configparser
import torch
import os
import torch.multiprocessing as mp


class ParamConfigurator:
    """Parameter configurator class for deep learning pipeline."""

    def __init__(self):
        """# TODO: Docstring"""
        config = configparser.ConfigParser()
        config.read('config.ini')

        # architecture
        self.architecture_type = config['architecture']['architecture_type']

        # training
        self.batch_size = config['training'].getint('batch_size')

        self.num_epochs = config['training'].getint('num_epochs')

        self.learning_rate = config['training'].getfloat('learning_rate')

        self.momentum = config['training'].getfloat('momentum')
        self.plot_frequency = config['training'].getint('plot_frequency')
        self.num_workers = config['training'].getint('num_workers')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':

            # Horovod: Import only if GPU is available
            import horovod.torch as hvd

            # Horovod: Initialize
            hvd.init()
            torch.manual_seed(42)

            # Horovod: Pin GPU to local rank.
            torch.cuda.set_device(hvd.local_rank())
            torch.cuda.manual_seed(42)

            # Horovod: Limit # of CPU threads to be used per worker.
            torch.set_num_threads(1)

            self.kwargs = {'num_workers': self.num_workers, 'pin_memory': True}

            # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
            # issues with Infiniband implementations that are not fork-safe
            if (self.kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                    mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
                self.kwargs['multiprocessing_context'] = 'forkserver'

            self.hvd_size = hvd.size()
            self.hvd_rank = hvd.rank()

        else:
            self.kwargs = {'num_workers': self.num_workers}

        self.optimizer = config['training']['optimizer']
        if self.optimizer not in ['Adam', 'SGD']:
            raise ValueError(f"[ERROR] Chosen optimizer is not valid! Please choose out of ['Adam, 'SGD].")

        self.loss_function = config['training']['loss_function']
        if self.loss_function == 'BCE':
            self.loss_function = torch.nn.BCELoss()

        # train_data
        # self.train_data_size = config['train_data'].getint('train_data_size')
        self.train_data_dir = config['train_data']['train_data_dir']
        if not os.path.exists(self.train_data_dir):
            raise ValueError(f"[ERROR] Directory specified for training data does not exist!")

        # validation_data
        # self.validation_data_size = config['validation_data'].getint('validation_data_size')
        self.validation_data_dir = config['validation_data']['validation_data_dir']
        if not os.path.exists(self.validation_data_dir):
            raise ValueError(f"[ERROR] Directory specified for validation data does not exist!")

        # ResNet
        self.resnet_depth = config['ResNet'].getint('depth')
        if self.resnet_depth not in [18, 50, 101, 152]:
            raise ValueError(f"[ERROR] ResNet is only available for depths of [18, 50, 101, 152].")

        # Inception
        self.inception_version = config['InceptionNet'].getint('version')
        if self.inception_version not in [int(1), int(3)]:
            raise ValueError(f"[ERROR] InceptionNet is only available for version 1 and for version 3.")
