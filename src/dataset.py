import os


class Dataset:
    def __init__(self, dir, transform=None, limit_files=None):
        self.limit_files = limit_files
        self.models = os.listdir(dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.models)


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
