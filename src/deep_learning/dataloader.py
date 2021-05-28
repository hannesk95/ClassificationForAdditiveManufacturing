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

    def __len__(self):
        return configuration.train_data_configuration.training_data_size

    def __getitem__(self, index):
        model = np.load("")['model']
        label = np.load("")['label']

        if self.transform is not None:
            model = torch.Tensor([model])
        
        return model,label
        