import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from config import BATCH_SIZE

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

class SteerPredModel(nn.Module):
    def __init__(self):
        super(SteerPredModel, self).__init__()

        self.img_f = nn.Sequential(OrderedDict({
                'conv1' : nn.Conv2d(3, 24, 7, 4),
                "maxpool1" : nn.MaxPool2d(5, 2),
                "batchnorm1:" : nn.BatchNorm2d(24),
                'relu1' : nn.ReLU(),

                'conv2' : nn.Conv2d(24, 36, 5, 3),
                "maxpool2" : nn.MaxPool2d(3, 1),
                "batchnorm2:" : nn.BatchNorm2d(36),
                'relu2' : nn.ReLU(),

                'conv3' : nn.Conv2d(36, 48, 5, 2),
                "maxpool3" : nn.MaxPool2d(3, 1),
                "batchnorm3:" : nn.BatchNorm2d(48),
                'relu3' : nn.ReLU(),

                'conv4' : nn.Conv2d(48, 256, 3, 2),
                "maxpool4" : nn.MaxPool2d(3, 1),
                "batchnorm4:" : nn.BatchNorm2d(256),
                'relu4' : nn.ReLU(),

                'conv5' : nn.Conv2d(256, 256, 1, 1),
                "batchnorm5:" : nn.BatchNorm2d(256),
                'relu5' : nn.ReLU(),
                }))

        self.img_l = nn.Sequential(OrderedDict({
                'conv1' : nn.Conv2d(3, 24, 7, 4),
                "maxpool1" : nn.MaxPool2d(5, 2),
                "batchnorm1:" : nn.BatchNorm2d(24),
                'relu1' : nn.ReLU(),

                'conv2' : nn.Conv2d(24, 36, 5, 3),
                "maxpool2" : nn.MaxPool2d(3, 1),
                "batchnorm2:" : nn.BatchNorm2d(36),
                'relu2' : nn.ReLU(),

                'conv3' : nn.Conv2d(36, 48, 5, 2),
                "maxpool3" : nn.MaxPool2d(3, 1),
                "batchnorm3:" : nn.BatchNorm2d(48),
                'relu3' : nn.ReLU(),

                'conv4' : nn.Conv2d(48, 256, 3, 2),
                "maxpool4" : nn.MaxPool2d(3, 1),
                "batchnorm4:" : nn.BatchNorm2d(256),
                'relu4' : nn.ReLU(),

                'conv5' : nn.Conv2d(256, 256, 1, 1),
                "batchnorm5:" : nn.BatchNorm2d(256),
                'relu5' : nn.ReLU(),
                }))

        self.img_r = nn.Sequential(OrderedDict({
                'conv1' : nn.Conv2d(3, 24, 7, 4),
                "maxpool1" : nn.MaxPool2d(5, 2),
                "batchnorm1:" : nn.BatchNorm2d(24),
                'relu1' : nn.ReLU(),

                'conv2' : nn.Conv2d(24, 36, 5, 3),
                "maxpool2" : nn.MaxPool2d(3, 1),
                "batchnorm2:" : nn.BatchNorm2d(36),
                'relu2' : nn.ReLU(),

                'conv3' : nn.Conv2d(36, 48, 5, 2),
                "maxpool3" : nn.MaxPool2d(3, 1),
                "batchnorm3:" : nn.BatchNorm2d(48),
                'relu3' : nn.ReLU(),

                'conv4' : nn.Conv2d(48, 256, 3, 2),
                "maxpool4" : nn.MaxPool2d(3, 1),
                "batchnorm4:" : nn.BatchNorm2d(256),
                'relu4' : nn.ReLU(),

                'conv5' : nn.Conv2d(256, 256, 1, 1),
                "batchnorm5:" : nn.BatchNorm2d(256),
                'relu5' : nn.ReLU(),
                }))

        self.fc = nn.Sequential(OrderedDict({
                'linear0' : nn.Linear(24576, 1024),
                'relu0' : nn.ReLU(),

                'linear1' : nn.Linear(1024, 512),
                'relu1' : nn.ReLU(),

                'linear2' : nn.Linear(512, 100),
                'relu2' : nn.ReLU(),

                'linear3' : nn.Linear(100, 50),
                'relu3' : nn.ReLU(),

                'linear4' : nn.Linear(50, 10),
                'relu4' : nn.ReLU(),

                'linear5' : nn.Linear(10, 7),
                }))

    def forward(self, x_img_f, x_img_l, x_img_r):
        x = self.img_f(x_img_f)
        x = x.view(x.size(0), -1)

        y = self.img_l(x_img_l)
        y = y.view(y.size(0), -1)

        z = self.img_r(x_img_r)
        z = z.view(z.size(0), -1)

        feat = torch.cat((x, y, z), dim=1)
        c = self.fc(feat)

        return c, feat
