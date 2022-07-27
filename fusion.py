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

class ThrottlePredModel(nn.Module):
    def __init__(self):
        super(ThrottlePredModel, self).__init__()
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

                nn.Linear(64, 1),
                nn.ReLU()
                )

    def forward(self, x_img):
        x = self.ImgFeatExtractor(x_img)
        x = self.fc(x)

        return x

class BrakePredModel(nn.Module):
    def __init__(self):
        super(BrakePredModel, self).__init__()
        self.ImgFeatExtractor = ImgFeatExtractor_2D()
        self.pointnet = PointNet()

        self.fc = nn.Sequential(
                nn.Linear(24576+1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),

                nn.Linear(64, 1),
                nn.ReLU()
                )

    def forward(self, x_img, x_pts):
        x = self.ImgFeatExtractor(x_img)
        y, _, _ = self.pointnet(x_pts)
        z =torch.cat((x, y), dim=1)
        z = self.fc(z)

        return z

class SteerPredModel(nn.Module):
    def __init__(self):
        super(SteerPredModel, self).__init__()
        self.pointnet = PointNet()

        self.seq = nn.Sequential(
                nn.Conv2d(3, 24, 5, 2),
                nn.ReLU(),

                nn.Conv2d(24, 36, 5, 2),
                nn.ReLU(),
                
                nn.Conv2d(36, 48, 5, 2),
                nn.ReLU(),

                nn.Conv2d(48, 64, 3, 1),
                nn.ReLU(),

                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU(),
                
                nn.Dropout(0.5)
                )

        self.fc = nn.Sequential(
                nn.Linear(247616+1024, 100),
                nn.ReLU(),

                nn.Linear(100, 50),
                nn.ReLU(),

                nn.Linear(50, 10),
                nn.ReLU(),
                
                nn.Linear(10, 1),
                )

    def forward(self, x_img, x_pts):
        x = self.seq(x_img)
        x = x.reshape([x_img.shape[0], -1])
        y, _, _ = self.pointnet(x_pts)
        z = torch.cat((x, y), dim=1)
        z = self.fc(z)

        return z
