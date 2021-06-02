import torch
from src.deep_learning.network import Vanilla3DCNN, ResNet, VGGNet, InceptionNet_v1, InceptionNet_v3


class ArchitectureSelector:
    """# TODO: Docstring"""

    def __init__(self, nn_architecture: str, config):
        """# TODO: Docstring"""

        self.nn_architecture = nn_architecture
        self.architectures = ["Vanilla3DCNN", "ResNet", "VGGNet", "InceptionNet"]
        self.config = config
        self.model = None

    def select_architecture(self):
        """# TODO: Docstring"""

        if self.nn_architecture not in self.architectures:
            raise ValueError(f"Chosen neural network architecture is not valid! Choose out of {self.architectures}")

        if self.nn_architecture == "Vanilla3DCNN":
            self.model = Vanilla3DCNN()

        if self.nn_architecture == "ResNet":
            self.model = ResNet.generate_model(self.config.resnet_depth)

        if self.nn_architecture == "VGGNet":
            self.model = VGGNet()

        if self.nn_architecture == "InceptionNet":
            if self.config.inception_version == 1:
                self.model = InceptionNet_v1()
            else:
                self.model = InceptionNet_v3()

        if self.config.optimizer == 'Adam':
            self.config.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        if self.config.optimizer == 'SGD':
            self.config.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)

        return self.model
