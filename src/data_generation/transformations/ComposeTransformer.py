class ComposeTransformer:
    """Transform class that combines multiple other transforms into one"""
    def __init__(self, transforms: list) -> object:
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, model):
        """# TODO"""
        for transform in self.transforms:
            model = transform(model)
        return model
