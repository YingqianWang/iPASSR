from PIL import Image
import os
from torch.utils.data.dataset import Dataset
import random
import torch
import numpy as np


class TrainSetLoader(Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = cfg.trainset_dir + '/patches_x' + str(cfg.scale_factor)
        self.file_list = os.listdir(self.dataset_dir)
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr0.png')
        img_hr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr1.png')
        img_lr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr0.png')
        img_lr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr1.png')
        img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left  = np.array(img_lr_left,  dtype=np.float32)
        img_lr_right = np.array(img_lr_right, dtype=np.float32)
        img_hr_left, img_hr_right, img_lr_left, img_lr_right = augmentation(img_hr_left, img_hr_right, img_lr_left, img_lr_right)
        return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right)

    def __len__(self):
        return len(self.file_list)


def augmentation(hr_image_left, hr_image_right, lr_image_left, lr_image_right):
        if random.random()<0.5:     # flip horizonly
            lr_image_left_ = lr_image_right[:, ::-1, :]
            lr_image_right_ = lr_image_left[:, ::-1, :]
            hr_image_left_ = hr_image_right[:, ::-1, :]
            hr_image_right_ = hr_image_left[:, ::-1, :]
            lr_image_left, lr_image_right = lr_image_left_, lr_image_right_
            hr_image_left, hr_image_right = hr_image_left_, hr_image_right_

        if random.random()<0.5:     #flip vertically
            lr_image_left = lr_image_left[::-1, :, :]
            lr_image_right = lr_image_right[::-1, :, :]
            hr_image_left = hr_image_left[::-1, :, :]
            hr_image_right = hr_image_right[::-1, :, :]

        return np.ascontiguousarray(hr_image_left), np.ascontiguousarray(hr_image_right), \
                np.ascontiguousarray(lr_image_left), np.ascontiguousarray(lr_image_right)

def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)
