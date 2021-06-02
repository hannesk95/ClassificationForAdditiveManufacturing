import os
import numpy as np
from torch.utils.data import Dataset
import torch


class AMCDataset(Dataset):
    """Additive Manufacturing Classification (AMC) dataset."""

    def __init__(self, root_dir: str, transform=None, cutoff: int = None):
        """# TODO: Docstring"""
        super(AMCDataset, self).__init__()
        self.root_dir = root_dir
        self.cutoff = cutoff
        self.models = self._load_model_paths()
        self.transform = transform

    def __len__(self):
        """# TODO: Docstring"""
        return len(self.models)

    def _load_model_paths(self) -> list:
        """# TODO: Docstring"""
        models = os.listdir(self.root_dir)
        models = [elem for elem in models if elem.endswith('.npz')]
        if self.cutoff is not None:
            models = models[:self.cutoff]
        models = [os.path.join(self.root_dir, model_path) for model_path in models]
        return models

    def __getitem__(self, idx):
        """# TODO: Docstring"""
        model = np.load(self.models[idx])['model']
        label = np.load(self.models[idx])['label']

        if self.transform:
            model = self.transform(model)

        sample = {'model': torch.from_numpy(model), 'label': torch.from_numpy(label)}
        return sample
