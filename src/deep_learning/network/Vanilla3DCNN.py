import torch
import torch.nn as nn
from torchsummary import summary


class Vanilla3DCNN(nn.Module):
    """#TODO: Add docstring."""

    def __init__(self):
        """#TODO: Add docstring."""

        super(Vanilla3DCNN, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(9, 9, 9))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(7, 7, 7))
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=96, kernel_size=(5, 5, 5))
        self.conv4 = nn.Conv3d(in_channels=96, out_channels=128, kernel_size=(3, 3, 3))

        self.fc1 = nn.Linear(in_features=256, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.max_pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, x):
        """#TODO: Add docstring."""

        # Convolution block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool3d(x)

        # Convolution block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool3d(x)

        # Convolution block 3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool3d(x)

        # Convolution block 4
        x = self.conv4(x)
        x = self.relu(x)
        x = self.max_pool3d(x)

        # Transition block to fully connected layers
        x = nn.AvgPool3d(kernel_size=2, stride=1)(x)
        x = nn.MaxPool3d(kernel_size=x.shape[-1])(x)
        x = torch.flatten(x)
        x = self.dropout(x)

        # Classification block (fully connected)
        # x = nn.Linear(x.shape[0], 32)(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x = nn.Linear(32, 1)(x).cuda()
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


# For testing purposes
if __name__ == '__main__':

    model = Vanilla3DCNN()

    if torch.cuda.device_count() > 0:
        summary(model.cuda(), (1, 128, 128, 128))
    else:
        summary(model, (1, 128, 128, 128))
