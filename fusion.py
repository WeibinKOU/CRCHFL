import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import BATCH_SIZE, THREED_CHANNEL

from pointnet import PointNetfeat as PointNet
from resnet50 import ResNet50 

class ImgFeatExtractor_2D(nn.Module):
    def __init__(self):
        super(ImgFeatExtractor_2D, self).__init__()
        self.feat_extractor = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(3, 2),

                nn.Conv2d(64, 128, 5, 2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(3, 2),

                nn.Dropout(0.3),

                nn.Conv2d(128, 256, 3, 2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(3, 2),

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
                nn.Conv3d(3, 32, (2, 5, 5), (1, 3, 3)),
                nn.BatchNorm3d(32), 
                nn.LeakyReLU(0.1),
                nn.MaxPool3d(3, 1),

                nn.Conv3d(32, 64, (2, 3, 3), (1, 2, 2)),
                nn.BatchNorm3d(64), 
                nn.LeakyReLU(0.1),
                nn.MaxPool3d((2, 5, 5), (1, 2, 2)),

                nn.Dropout(0.3),

                nn.Conv3d(64, 128, (2, 2, 2), 1),
                nn.BatchNorm3d(128), 
                nn.LeakyReLU(0.1),
                nn.MaxPool3d((2, 5, 5), (1, 2, 2)),
                )

    def forward(self, x_vol):
        x = self.feat_extractor(x_vol)
        x = x.reshape([x_vol.shape[0], -1])
        return x

class ThrottleBrakePredModel(nn.Module):
    def __init__(self):
        super(ThrottleBrakePredModel, self).__init__()
        self.ImgFeatExtractor = ImgFeatExtractor_2D()

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

    def forward(self, x_img):
        x = self.ImgFeatExtractor(x_img)
        x = self.fc(x)

        return x

class SteerPredModel(nn.Module):
    def __init__(self):
        super(SteerPredModel, self).__init__()
        self.VolFeatExtractor = ImgFeatExtractor_3D()

        self.vfc = nn.Sequential(
                nn.Linear(47104, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1),
                
                nn.Linear(64, 1),
                nn.Tanh()
                )

    def forward(self, x_vol):
        x = self.VolFeatExtractor(x_vol)
        x = self.vfc(x)

        return x
