import torch
import horovod.torch as hvd
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

        if self.config.device.type == 'cuda':

            # Send network to GPU if available
            self.model.cuda()

            # Initialize Horovod
            hvd.init()

            # Pin GPU to be used to process local rank (one GPU per process)
            torch.cuda.set_device(hvd.local_rank())

            # Store size and rank into config parameters
            self.config.hvd_size = hvd.size()
            self.config.hvd_rank = hvd.rank()

        # Define optimizer
        if self.config.optimizer == 'Adam':
            self.config.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        if self.config.optimizer == 'SGD':
            self.config.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)

        if self.config.device.type == 'cuda':

            # Add Horovod Distributed Optimizer
            self.config.optimizer = hvd.DistributedOptimizer(self.config.optimizer,
                                                             named_parameters=self.model.named_parameters())

            # Broadcast parameters from rank 0 to all other processes.
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            # hvd.broadcast_optimizer_state(self.config.optimizer, root_rank=0)

        return self.model
