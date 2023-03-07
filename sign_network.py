import torch
import torch.nn as nn
import torchvision.models as models

class Signmodel(nn.Module):
    def __init__(self, no_classes):
        super(Signmodel, self).__init__()
        self.conv2d = models.resnet18(pretrained=True)
        out_of_resnet = self.conv2d.fc.in_features
        self.conv2d.fc = nn.Linear(in_features=out_of_resnet, out_features=no_classes)

    def forward(self, x):
        out = self.conv2d(x)
        return out

