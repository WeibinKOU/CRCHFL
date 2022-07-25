import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pointnet import PointNetfeat as PointNet
from resnet50 import ResNet50 

class ImgFeatExtractor_2D(nn.Module):
    def __init__(self):
        super(ImgFeatExtractor_2D, self).__init__()
        self.process_rgb = nn.Sequential(
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
        x = self.process_rgb(x_rgb)
        x = x.reshape((x.shape[0], -1))

        return x

class ThrottleBrakeModel(nn.Module):
    def __init__(self):
        super(ThrottleBrakeModel, self).__init__()
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

    def forward(self, x_rgb):
        x = self.ImgFeatExtractor(x_rgb)
        x = self.fc(x)

        return x

