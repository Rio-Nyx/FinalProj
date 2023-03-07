import glob

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from basicfunc import *

class SignDataset(Dataset):
    def __init__(self, mode):
        self.inputs_list = np.load(f"preprocess/phoenix2014/{mode}_info.npy", allow_pickle=True).item()
        easyprint('saved dictionary of all details', self.inputs_list)
        self.dict = np.load(f"preprocess/phoenix2014/gloss_dict.npy", allow_pickle=True).item()
        easyprint('gloss dict', self.dict)

    def __getitem__(self, index):
        fi = self.inputs_list[index] # input_list is the dict // its printed.. see
        path = 'dataset/phoenix-2014-multisigner/features/fullFrame-256x256px/' + fi['folder']
        images_path_list = glob.glob(path)
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        sign_images = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in images_path_list]
        normalized_img = [sign_pic/255.0 for sign_pic in sign_images]  # this is list of all images for that sign
        return normalized_img, label_list


    def __len__(self):
        return len(self.inputs_list) - 1   # -1 is to avoid the first prefix entry


    def test_output(self, index):
        '''
        this function is just implemented to test ie. debugging

        :param index:
        :return:
        '''

        fi = self.inputs_list[index]  # input_list is the dict // its printed.. see
        path = 'dataset/phoenix-2014-multisigner/features/fullFrame-256x256px/' + fi['folder']
        images_path_list = glob.glob(path)
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        sign_images = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in images_path_list]
        # normalized_img = [cv2.normalize(sign_pic, None, 0, 1, cv2.NORM_MINMAX) for sign_pic in sign_images]
        normalized_img = [sign_pic/255.0 for sign_pic in sign_images]  # this is list of all images for that sign
        return normalized_img, label_list

