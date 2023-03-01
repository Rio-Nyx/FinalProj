import os

import cv2
import numpy as np
import torch
from cv2 import data
from setuptools import glob

class BaseFeeder(data.Dataset):
    def __init__(self, input_list, prefix, gloss_dict):
        self.input_list = input_list
        self.prefix = prefix
        self.dict = gloss_dict

    def __getitem__(self, idx):
        input_data, label, fi = self.read_video(idx)
        # TODO: normalize the data
        input_data, label = self.normalize(input_data, label)
        return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']

    def __len__(self):
        return len(self.inputs_list) - 1

    def read_video(self, index, num_glosses=-1):
        # load file info
        fi = self.inputs_list[index]
        img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'])
        img_list = sorted(glob.glob(img_folder))
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0]) # we are appending the unique index to it
        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], label_list, fi

    def normalize(self, video, label, file_id=None):
        if isinstance(video, list):
            video = np.array(video)
            video = torch.from_numpy(video.transpose((0, 3, 1, 2))).float()
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose((0, 3, 1, 2)))
            video = video.float() / 127.5 - 1
        return video, label