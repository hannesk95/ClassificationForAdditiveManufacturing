import os
import torch
import numpy as np
from torch.utils.data import Dataset


class AMCDataset(Dataset):
    """Additive Manufacturing Classification (AMC) dataset."""

    def __init__(self, config: object, transform=None, cutoff: int = None):
        """# TODO: Docstring"""
        super(AMCDataset, self).__init__()
        self.config = config
        self.data_dir = self.config.data_dir
        self.cutoff = self.config.cutoff
        self.models = self._load_model_paths()
        self.transform = transform

    def __len__(self):
        """# TODO: Docstring"""
        return len(self.models)

    def _load_model_paths(self) -> list:
        """# TODO: Docstring"""
        models = os.listdir(self.data_dir)
        self.config.data_len = len(models)
        models = [elem for elem in models if elem.endswith('.npz')]
        if self.cutoff is not 0:
            models = models[:self.cutoff]
        models = [os.path.join(self.data_dir, model_path) for model_path in models]
        return models

    def __getitem__(self, idx):
        """# TODO: Docstring"""
        model = np.load(self.models[idx])['model']
        label = torch.tensor(np.load(self.models[idx])['label'], dtype=torch.float32)

        if self.transform:
            model = self.transform(model)

        model = torch.reshape(model, (1, model.shape[0], model.shape[1], model.shape[2]))
        model = model.to(torch.float32)

        return model, label
