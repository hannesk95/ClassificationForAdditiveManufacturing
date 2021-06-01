import torch
import torch.nn as nn
from torchsummary import summary


class Vanilla3DCNN(nn.Module):
    """#TODO: Add docstring."""

    def __init__(self):
        """#TODO: Add docstring."""

        super(Vanilla3DCNN, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(9, 9, 9))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(7, 7, 7))
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(5, 5, 5))
        self.fc1 = nn.Linear(46656, 64)
        self.fc2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.dropout3d = nn.Dropout3d(p=0.2)
        self.max_pool3d = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """#TODO: Add docstring."""

        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.conv2(x)
        x = self.max_pool3d(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.conv3(x)
        x = self.max_pool3d(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


# For testing purposes
if __name__ == '__main__':

    model = Vanilla3DCNN()

    if torch.cuda.device_count() > 0:
        summary(model.cuda(), (1, 64, 64, 64))
    else:
        summary(model, (1, 64, 64, 64))
