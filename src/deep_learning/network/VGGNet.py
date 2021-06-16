import torch
import torch.nn as nn
from torchsummary import summary


def conv_3x3x3(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, bias=False)


class VGGNet(nn.Module):
    def __init__(self, in_channels=1):
        super(VGGNet, self).__init__()
        self.conv1 = conv_3x3x3(in_planes=in_channels, out_planes=32, kernel_size=5, stride=2)
        self.batchnorm1 = nn.BatchNorm3d(32)
        self.maxpool1 = nn.MaxPool3d(kernel_size=3, stride=2)
        self.conv2 = conv_3x3x3(in_planes=32, out_planes=64, stride=2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.batchnorm2 = nn. BatchNorm3d(64)
        self.conv3 = conv_3x3x3(in_planes=64, out_planes=128,stride=2)
        self.batchnorm3 = nn. BatchNorm3d(128)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout3d(p = 0.4)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.maxpool2(x)
    
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x


# Testing
if __name__ == "__main__":
    net = VGGNet()
    summary(net, (1, 128, 128, 128))
