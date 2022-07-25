import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pointnet import PointNetfeat as PointNet
from resnet50 import ResNet50 

class Fusion(nn.Module):
    def __init__(self, img_only, lidar_only, both):
        super(Fusion, self).__init__()
        self.img_only = img_only
        self.lidar_only = lidar_only
        self.both = both 

        lens = 0
        if img_only:
            lens = 784 
            #lens = 78400 
        elif lidar_only:
            lens = 1024 
        elif both:
            lens = 79424 

        self.pointnet = PointNet(global_feat = True,
                feature_transform = False) 

        if self.both:
            self.fusion = nn.Sequential(
                    nn.Linear(lens, 1024),
                    nn.LeakyReLU(0.3),

                    nn.Linear(1024, 1024),
                    nn.MaxPool1d(2, 2),
                    nn.LeakyReLU(0.3),
                    nn.Dropout(0.3),

                    nn.Linear(512, 1024),
                    nn.LeakyReLU(0.3),
                    )

        if self.img_only:
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


    def forward(self, x_rgb, x_pts):
        if self.both:
            x_res = self.process_rgb(x_rgb)
            x_res = x_res.reshape((x_res.shape[0], -1))

            x_pts = self.pointnet(x_pts)
            x = torch.cat((x_res, x_pts[0]), dim = 1)
            x = self.fusion(x)
            x = x.reshape((x.shape[0], -1))
        elif self.img_only:
            x = self.process_rgb(x_rgb)
            x = x.reshape((x.shape[0], -1))
        elif self.lidar_only:
            x = self.pointnet(x_pts)[0]

        return x

class ActionPredictModel(nn.Module):
    def __init__(self, img_only, lidar_only, both):
        super(ActionPredictModel, self).__init__()
        self.Fusion = Fusion(img_only, lidar_only, both)
        lens = 0
        if img_only:
            lens = 24576 
            #lens = 9216 
        elif lidar_only:
            lens = 1024
        elif both:
            lens = 1024 

        self.fc = nn.Sequential(
                nn.Linear(lens, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),

                nn.Linear(64, 3),
                )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_rgb, x_pts):
        x = self.Fusion(x_rgb, x_pts)

        x = self.fc(x)
        x[:, 0:2] = self.relu(x[:, 0:2])
        x[:, 2] = self.sigmoid(x[:, 2])

        return x

