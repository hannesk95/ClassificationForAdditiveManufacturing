from torchvision import models
import torch
import torch.nn as nn
from torchsummary import summary

class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()

    def forward(self,x):
        return x

model = models.video.r3d_18(pretrained=True)
model.stem[0] = nn.Sequential(nn.Conv3d(1,64,(3,7,7),(1,2,2),(1,3,3,),bias=False), nn.BatchNorm3d(64),nn.ReLU())
model.fc = nn.Linear(512,10)
summary(model,(1,128,128,128))