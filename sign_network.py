import torch
import torch.nn as nn
import torchvision.models as models

class Signmodel(nn.Module):
    def __init__(self, no_classes):
        super(Signmodel, self).__init__()
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.conv2d = models.resnet18(pretrained=True)
        out_of_resnet = self.conv2d.fc.in_features
        self.conv2d.fc = nn.Linear(in_features=out_of_resnet, out_features=no_classes)

    def forward(self, x, x_lgt, label=label, label_lgt=label_lgt):
        out = self.conv2d(x)
        # TODO : out.permute(2,0,1)
        return {
            "conv_out" : out,
            "conv_len" : x_lgt
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        # loss of lstm only need to be considered at end
        weight = 1
        loss = weight * self.loss['CTCLoss'](ret_dict["conv_out"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["conv_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
