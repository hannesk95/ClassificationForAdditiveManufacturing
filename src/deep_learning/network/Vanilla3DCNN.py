import torch
import torch.nn as nn
from torchsummary import summary


class Vanilla3DCNN(nn.Module):
    """#TODO: Add docstring."""

    # def __init__(self):
    #     """#TODO: Add docstring."""
    #
    #     super(Vanilla3DCNN, self).__init__()
    #
    #     self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(9, 9, 9))
    #     self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(7, 7, 7))
    #     self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(5, 5, 5))
    #     self.fc1 = nn.Linear(64, 64)
    #     self.fc2 = nn.Linear(64, 1)
    #
    #     self.relu = nn.ReLU()
    #     self.dropout = nn.Dropout(p=0.2)
    #     self.dropout3d = nn.Dropout3d(p=0.2)
    #     self.max_pool3d = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2)
    #     self.sigmoid = nn.Sigmoid()
    #
    # def forward(self, x):
    #     """#TODO: Add docstring."""
    #
    #     x = self.conv1(x)
    #     x = self.relu(x)
    #     x = self.dropout3d(x)
    #     x = self.conv2(x)
    #     x = self.max_pool3d(x)
    #     x = self.relu(x)
    #     x = self.dropout3d(x)
    #     x = self.conv3(x)
    #     x = self.max_pool3d(x)
    #     x = self.relu(x)
    #     x = self.dropout3d(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc1(x)
    #     x = self.relu(x)
    #     x = self.dropout(x)
    #     x = self.fc2(x)
    #     x = self.sigmoid(x)
    #
    #     return x

    def __init__(self):
        super(Vanilla3DCNN, self).__init__()

        self.conv_layer1 = self._conv_layer_set(3, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(2 ** 3 * 64, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.LeakyReLU()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.15)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out


# For testing purposes
if __name__ == '__main__':

    model = Vanilla3DCNN()

    if torch.cuda.device_count() > 0:
        summary(model.cuda(), (1, 64, 64, 64))
    else:
        summary(model, (1, 64, 64, 64))
