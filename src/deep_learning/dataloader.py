from numpy.lib.utils import source
import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset
import configuration

class Data(Dataset):
    def __init__(self,transform=None):
        super(Data,self).__init__()
        self.transform = transform
        self.model_data = self.load_model_path()

    def __len__(self):
        return configuration.train_data_configuration.training_data_size

    def load_model_path(self):
        source_dir = configuration.train_data_configuration.training_set_dir
        path, dirs, files = next(os.walk(source_dir))
        files = sorted(files)

        models = self.model_initiliazer()

        for i in range(len(files)):
            data = np.load(source_dir + files[i])['model']
            label = np.load(source_dir + files[i])['label']
            models[i].append((data,label))

        print("Data loading complete")
        return models

    def model_initiliazer(self):
        model_data = {}
        source_dir = configuration.train_data_configuration.training_set_dir
        path, dirs, files = next(os.walk(source_dir))
        files = sorted(files)

        for i in range(len(files)):
            model_data[i] = []
        return model_data

    def __getitem__(self, index):
        source_dir = configuration.train_data_configuration.training_set_dir
        path, dirs, files = next(os.walk(source_dir))
        files = sorted(files)

        id = random.randint(0,len(files))
        data, label = random.choice(self.model_data[id])

        if self.transform is not None:
            data = torch.Tensor([data])
        
        return data,label
        