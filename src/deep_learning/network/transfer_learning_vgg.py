from torchvision import models
import torch
from torchsummary import summary

res = models.video.r3d_18(pretrained=True)
res.load_state_dict(torch.load("/home/aditya/.cache/torch/hub/checkpoints/r3d_18-b3b3357e.pth "))

for param in res.features.parameters():
    param.requires_grad = False

num_features = res.