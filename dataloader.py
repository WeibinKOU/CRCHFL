import torch
from torchvision import transforms
from glob import glob

from torch.utils.data import DataLoader, Dataset
from config import Tensor

import cv2
import numpy as np

class ImgData(Dataset):
    def __init__(self, dataset_dir, aug, transform):
        super(ImgData, self).__init__()
        self.transform = transform
        self.aug = aug
 
        self.imgs = glob(dataset_dir + "/*.png")
        name_list = []
        for img in self.imgs:
            name_list.append(img.split('\\')[-1].split('.')[0])

        self.name = name_list

    def __len__(self):
        return len(self.imgs) 

    def __getitem__(self, index):
        names = self.name[index]
        img = cv2.imread(self.imgs[index], cv2.IMREAD_COLOR)[..., ::-1]
        if self.aug is not None:
            img = self.aug(image=img)
        img = self.transform(img.copy())
        return names, img
         
class MultiImgData(Dataset):
    def __init__(self, root_dir, aug, transform):
        super(MultiImgData, self).__init__()
        self.transform = transform
        self.aug = aug

        self.imgs_f = glob(root_dir + "/imgs_f/" + "/*.png")
        self.imgs_l = glob(root_dir + "/imgs_l/" + "/*.png")
        self.imgs_r = glob(root_dir + "/imgs_r/" + "/*.png")
        name_list = []
        for img in self.imgs_f:
            name_list.append(img.split('\\')[-1].split('.')[0])

        self.name = name_list

    def __len__(self):
        return len(self.imgs_f)

    def __getitem__(self, index):
        names = self.name[index]
        imgs_f = cv2.imread(self.imgs_f[index], cv2.IMREAD_COLOR)[..., ::-1]
        imgs_l = cv2.imread(self.imgs_l[index], cv2.IMREAD_COLOR)[..., ::-1]
        imgs_r = cv2.imread(self.imgs_r[index], cv2.IMREAD_COLOR)[..., ::-1]
        if self.aug is not None:
            imgs_f = self.aug(image=imgs_f)
            imgs_l = self.aug(image=imgs_l)
            imgs_r = self.aug(image=imgs_r)
        imgs_f = self.transform(imgs_f.copy())
        imgs_l = self.transform(imgs_l.copy())
        imgs_r = self.transform(imgs_r.copy())
        return names, imgs_f, imgs_l, imgs_r

class ActionData():
    def __init__(self, action_file):
        self.data = np.load(action_file, allow_pickle = True)[()]

    def len(self):
        return len(self.data.items())

    def getitems(self, index_list):
        action_list = [] 
        for key in index_list:
            action_list.append(self.data[key])
        return np.array(action_list)

class LidarData():
    def __init__(self, lidar_dir):
        lidars = glob(lidar_dir + "/*.txt")
        lidar_dict = {}
        for lidar in lidars:
            lidar_dict[lidar.split('\\')[-1].split('.')[0]] = np.loadtxt(lidar)
        def align_lidar(lidar_dict):
            max_npoints = 0
            n_list = []
            for v in lidar_dict.values():
                n_list.append(v.shape[0])
            max_npoints = max(n_list)
            for k, v in lidar_dict.items():
                add = np.repeat(v[-1].reshape(1, 4), max_npoints - v.shape[0], axis=0)
                lidar_dict[k] = np.concatenate((v, add), axis=0)
        align_lidar(lidar_dict)
        self.data = lidar_dict 

    def len(self):
        return self.data.shape[0]

    def getitems(self, index_list):
        lidar_list = [] 
        for key in index_list:
            lidar_list.append(self.data[key][:, 0:3])
        return Tensor(np.array(lidar_list)).transpose(1, 2)
