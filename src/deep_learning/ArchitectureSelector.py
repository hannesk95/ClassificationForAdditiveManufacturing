import torch
from src.deep_learning.network import Vanilla3DCNN, ResNet, VGGNet, InceptionNet_v1, InceptionNet_v3, Vanilla3DCNN_large,Resnet_small


class ArchitectureSelector:
    """# TODO: Docstring"""

    def __init__(self, config: object):
        """# TODO: Docstring"""

        self.config = config
        # self.nn_architecture = self.config.architecture_type
        self.architectures = ["Vanilla3DCNN", "ResNet", "VGGNet", "InceptionNet", "Vanilla3DCNN_large", "Resnet_small"]
        self.model = None

    def select_architecture(self):
        """# TODO: Docstring"""

        if self.config.architecture_type not in self.architectures:
            raise ValueError(f"Chosen neural network architecture is not valid! Choose out of {self.architectures}")

        if self.config.architecture_type == "Vanilla3DCNN":
            self.model = Vanilla3DCNN()

        if self.config.architecture_type == "Vanilla3DCNN_large":
            self.model = Vanilla3DCNN_large()

        if self.config.architecture_type == "ResNet":
            self.model = ResNet.generate_model(self.config.resnet_depth, self.config.resnet_pretrained)
            self.config.experiment_name = self.config.architecture_type + str(self.config.resnet_depth) + "_pretrained_" + str(self.config.resnet_pretrained)
        
        if self.config.architecture_type == "Resnet_small":
            self.model = Resnet_small.generate_model()

        if self.config.architecture_type == "VGGNet":
            self.model = VGGNet()

        if self.config.architecture_type == "InceptionNet":
            if self.config.inception_version == 1:
                self.model = InceptionNet_v1()
            else:
                self.model = InceptionNet_v3()
            self.config.experiment_name = self.config.architecture_type + str(self.config.inception_version)

        # if self.config.device.type == 'cuda':

            # Send network to GPU if available
            # self.model.cuda()

        # Define optimizer
        if self.config.optimizer == 'Adam':
            self.config.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        if self.config.optimizer == 'SGD':
            self.config.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate,
                                                    momentum=self.config.momentum)

        return self.model
