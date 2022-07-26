import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import BATCH_SIZE

from pointnet import PointNetfeat as PointNet
from resnet50 import ResNet50 

class ImgFeatExtractor_2D(nn.Module):
    def __init__(self):
        super(ImgFeatExtractor_2D, self).__init__()
        self.feat_extractor = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2),
                nn.MaxPool2d(3, 2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1),

                nn.Conv2d(64, 128, 5, 2),
                nn.MaxPool2d(3, 2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),

                nn.Dropout(0.3),

                nn.Conv2d(128, 256, 3, 2),
                nn.MaxPool2d(3, 2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),

                nn.Conv2d(256, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                )


    def forward(self, x_rgb):
        x = self.feat_extractor(x_rgb)
        x = x.reshape((x.shape[0], -1))

        return x

class ImgFeatExtractor_3D(nn.Module):
    def __init__(self):
        super(ImgFeatExtractor_3D, self).__init__()
        self.feat_extractor = nn.Sequential(
                nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2)),
                nn.MaxPool3d(3, 2),
                nn.BatchNorm3d(32), 
                nn.LeakyReLU(0.1),

                nn.Conv3d(32, 64, 3, 1),
                nn.MaxPool3d((2, 3, 3), (1, 2, 2)),
                nn.BatchNorm3d(64), 
                nn.LeakyReLU(0.1),

                nn.Dropout(0.3),

                nn.Conv3d(64, 128, (1, 2, 2), 1),
                nn.MaxPool3d((2, 3, 3), (1, 2, 2)),
                nn.BatchNorm3d(128), 
                nn.LeakyReLU(0.1),

                #nn.Conv3d(128, 256, 1, 1),
                #nn.MaxPool3d(3, 2),
                #nn.BatchNorm3d(256), 
                #nn.LeakyReLU(0.1)
                )

    def forward(self, x_vol):
        x = self.feat_extractor(x_vol)
        x = torch.squeeze(x)
        x = x.reshape([BATCH_SIZE, -1])
        return x

class ActionPredModel(nn.Module):
    def __init__(self):
        super(ActionPredModel, self).__init__()
        self.ImgFeatExtractor = ImgFeatExtractor_2D()
        self.VolFeatExtractor = ImgFeatExtractor_3D()

        self.fc = nn.Sequential(
                nn.Linear(24576, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),

                nn.Linear(64, 2),
                nn.ReLU()
                )
        self.vfc = nn.Sequential(
                nn.Linear(39960, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                
                nn.Linear(64, BATCH_SIZE),
                nn.BatchNorm1d(BATCH_SIZE),
                nn.ReLU(),

                nn.Linear(BATCH_SIZE, 1),
                nn.BatchNorm1d(1),
                nn.ReLU()
                )

    def forward(self, x_img, x_vol):
        x = self.ImgFeatExtractor(x_img)
        x = self.fc(x)

        y = self.VolFeatExtractor(x_vol)
        y = self.vfc(y)

        z = torch.cat([x, y], dim=1)
        z[:, [0,1,2]] = z[:, [0,2,1]]

        return z 
