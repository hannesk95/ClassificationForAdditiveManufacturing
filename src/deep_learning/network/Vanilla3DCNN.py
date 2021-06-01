import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Vanilla3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Vanilla3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(9, 9, 9))
        self.conv2 = nn.Conv3d(32, 32, kernel_size=(7, 7, 7))
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(5, 5, 5))
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.1)
        self.dropout3d = nn.Dropout3d(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x, p=0.2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.2)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.2)
        x = x.view(-1, 3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.sigmoid(x, dim=1)


if __name__ == '__main__':

    model = Vanilla3DCNN()

    if torch.cuda.device_count() > 0:
        summary(model.cuda(), (1, 64, 64, 64))
    else:
        summary(model, (1, 64, 64, 64))