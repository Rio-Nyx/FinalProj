import os
import torch.optim as optim

from utility.parameters import get_parser
from utility.device import GpuDataParallel
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pdb
import sys
import cv2
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from slr_network import SLRModel
# from dataset.dataloader_video import BaseFeeder
faulthandler.enable()

class Processor:
    def __init__(self, arg):
        self.arg = arg
        self.device = GpuDataParallel()
        self.dataset = {}
        self.data_loader = {}

        # gloss dict we get after preprocessing in which each word has its associated integer_index and occurance, its a list [uniq_index, occourance]
        # dataset_info is specifically being loaded from main method
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()

        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        print('total no.of classes : ', arg.model_args['num_classes'])

        self.model, self.optimizer = self.loading()


    def loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        model = SLRModel(
            **self.arg.model_args,
            num_classes= self.arg.model_args['num_classes'],
            c2d_type= self.arg.model_args['c2d_type'],
            conv_type=2,
            use_bn=1,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
        )
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.arg.optimizer_args['base_lr'],
            weight_decay=self.arg.optimizer_args['weight_decay']
        )

        return model, optimizer


    def start(self):
        print('started')
        # TODO : load data in required format
        # TODO : implement train function complete
        # TODO : implemnt eval func



if __name__ == '__main__':
    sparser = get_parser()
    p = sparser.parse_args()   # returns a argparse.ArgumentParser class
    p.config = "configs\\baseline.yaml"
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)

        sparser.set_defaults(**default_arg)

    args = sparser.parse_args()

    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")

    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    processor = Processor(args)
    processor.start()
    print("All finished")