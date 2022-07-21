import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pointnet import PointNetDenseCls as PointNet
from resnet50 import ResNet50 

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.resnet = ResNet50() 
        self.pointnet = PointNet() 
        self.fusion = nn.Sequential(
                nn.Conv2d(2176, 320, 3, 2),
                nn.BatchNorm2d(320),
                nn.ReLU(),
                #nn.Conv2d(1024, 256, 3, 2),
                #nn.BatchNorm2d(256),
                #nn.ReLU(),
                #nn.Conv2d(256, 128, 1, 1),
                #nn.BatchNorm2d(128),
                #nn.ReLU(),
                nn.Conv2d(320, 64, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
                )


    def forward(self, x_rgb, x_pts):
        x_res = self.resnet(x_rgb)
        x_pts = self.pointnet(x_pts)

        h = x_res.shape[2] if x_res.shape[2] > x_pts.shape[2] else x_pts.shape[2]
        w = x_res.shape[3] if x_res.shape[3] > x_pts.shape[3] else x_pts.shape[3]

        diffY = h - x_res.size()[2]
        diffX = w - x_res.size()[3]

        x_res = F.pad(x_res, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

        diffY = h - x_pts.size()[2]
        diffX = w - x_pts.size()[3]

        x_pts = F.pad(x_pts, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

        x = torch.cat((x_res, x_pts), dim = 1)
        x = self.fusion(x)
        x = x.reshape((x.shape[0], -1))
        #print("Flattend fusion feature size: ", x.shape)

        return x

class ActionPredictModel(nn.Module):
    def __init__(self):
        super(ActionPredictModel, self).__init__()
        self.Fusion = Fusion()
        self.throttle = nn.Sequential(
                nn.Linear(968192, 1),
                nn.ReLU()
                )
        self.steer = nn.Sequential(
                nn.Linear(968192, 1),
                nn.Tanh()
               )
        self.brake = nn.Sequential(
                nn.Linear(968192, 1),
                nn.Sigmoid()
                )
        self.reverse = nn.Sequential(
                nn.Linear(968192, 1),
                nn.Sigmoid()
                )

    def forward(self, x_rgb, x_pts):
        x = self.Fusion(x_rgb, x_pts)

        throttle = self.throttle(x)
        steer = self.steer(x)
        brake = self.brake(x)
        reverse = self.reverse(x)

        return throttle, steer, brake, reverse

