import torch
from torchvision import transforms
from glob import glob

from torch.utils.data import DataLoader, Dataset
from config import Tensor, MAX_PT_NUM

import cv2
import numpy as np

#data_transform = transforms.Compose([transforms.ToTensor(),
#    transforms.Resize([480, 640]),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class ImgData(Dataset):
    def __init__(self, dataset_dir, transform):
        super(ImgData, self).__init__()
        self.transform = transform
        self.imgs = glob(dataset_dir + "/*.png")
        #img_list = []
        name_list = []
        for img in self.imgs:
        #    img_list.append(np.array(transform(cv2.imread(img, cv2.IMREAD_COLOR))))
            name_list.append(img.split('\\')[-1].split('.')[0])

        #self.data = Tensor(np.array(img_list)) 
        self.name = name_list

        #action_raw = np.load(dataset_dir + "/action.npy", allow_pickle = True)[()]
        #action_eff = []
        #for i in range(0, action_raw.shape[0], 2):
        #    action_eff.append(np.mean(action_raw[i : (i+1), :], axis = 0))
        #self.action = Tensor(np.array(action_eff))

    def __len__(self):
        return len(self.imgs) 

    def __getitem__(self, index):
        return self.name[index], Tensor(np.array(self.transform(cv2.imread(self.imgs[index], cv2.IMREAD_COLOR))))
         

class ActionData():
    def __init__(self, action_file):
        self.data = np.load(action_file, allow_pickle = True)[()]

    def len(self):
        return len(self.data.items())

    def getitems(self, index_list):
        action_list = [] 
        for key in index_list:
            action_list.append(self.data[key])
        return Tensor(np.array(action_list))

#img_dataset = ImgData("./dataset/", data_transform)
#img_dataloader = DataLoader(img_dataset, 
#        batch_size=64,
#        shuffle=True,
#        num_workers=0,
#        drop_last=False)
    
#for name, imgs in tqdm(img_dataloader):
#    print(name)
#    print(imgs)

class LidarData():
    def __init__(self, lidar_dir):
        lidars = glob(lidar_dir + "/*.txt")
        lidar_dict = {}
        for lidar in lidars:
            lidar_dict[lidar.split('\\')[-1].split('.')[0]] = np.loadtxt(lidar)
        def align_lidar(lidar_dict):
            #max_npoints = 0
            #n_list = []
            #for v in lidar_dict.values():
            #    n_list.append(v.shape[0])
            #max_npoints = max(n_list)
            for k, v in lidar_dict.items():
                add = np.repeat(v[-1].reshape(1, 4), MAX_PT_NUM - v.shape[0], axis=0)
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

#keys= ['0000002419', '0000002412', '0000002459', '0000002495', '0000002444', '0000002507', '0000002443', '0000002406', '0000002416', '0000002464', '0000002438', '0000002520', '0000002410', '0000002488', '0000002473', '0000002436', '0000002501', '0000002470', '0000002523', '0000002475', '0000002487', '0000002489', '0000002453', '0000002452', '0000002494', '0000002408']
#keys= ['0000002523', '0000002475', '0000002487', '0000002489', '0000002453', '0000002452', '0000002494', '0000002408']

#lidar_dataset = LidarData("./dataset/lidars/")
#lidar = lidar_dataset.getitems(keys)

