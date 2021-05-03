import os
from stl import mesh
import numpy as np

from src.data_generation.utils import extract_file_name


class Dataset:
    def __init__(self, dir_path, target_path=None, transform=None, limit_files=None, defector=None):
        self.limit_files = limit_files
        models = os.listdir(dir_path)
        models = [elem for elem in models if elem.endswith('.stl')]
        self.models = [os.path.join(dir_path, elem) for elem in models]

        if target_path is None:
            self.target_path = os.path.join(dir_path, 'SyntheticDataset')
            if not os.path.exists(self.target_path):
                os.makedirs(self.target_path)
        else:
            self.target_path = target_path

        self.transform = transform
        self.defector = defector

    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):
        model_path = self.models[index]
        model_name = extract_file_name(model_path)
        model = mesh.Mesh.from_file(model_path)
        label = 1

        if self.transform is not None:
            model = self.transform(model)

        if self.defector is not None:
            model, label = self.defector(model)

        target_path = os.path.join(self.target_path, model_name)
        np.savez_compressed(target_path, model=model, label=label)
