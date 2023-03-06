import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class SLRNet(nn.Module):
    def __init__(self):
        super(SLRNet, self).__init__()
        self.conv2d = models.resnet18(pretrained = True)
        self.conv2d.fc = nn.Identity()


