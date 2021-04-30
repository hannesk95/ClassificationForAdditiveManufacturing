import os
from stl import mesh


class Dataset:
    def __init__(self, dir_path, transform=None, limit_files=None):
        self.limit_files = limit_files
        self.models = os.listdir(dir_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):
        model_path = self.models[index]
        model = mesh.Mesh.from_file(model_path)

        if self.transform is not None:
            model = self.transform(model)

        # Save model
        

class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""
    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images
