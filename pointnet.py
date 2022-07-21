from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from config import device


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 4, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.maxpool1 = nn.MaxPool1d(kernel_size = 7, stride = 5)
        self.maxpool2 = nn.MaxPool1d(kernel_size = 5, stride = 3)
        self.maxpool3 = nn.MaxPool1d(kernel_size = 3, stride = 2)

        self.conv2d1 = nn.Conv2d(3, 64, (2, 3), (1, 2))
        self.conv2d2 = nn.Conv2d(3, 64, (2, 3), (1, 2))
        self.conv2d3 = nn.Conv2d(3, 64, (2, 3), (1, 2))
        self.layers = nn.Sequential(
                nn.Conv2d(3, 64, (2, 4), (1, 3)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, (2, 4), (1, 3)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                #nn.Conv2d(128, 256, 1, (1, 2)),
                #nn.BatchNorm2d(256),
                #nn.ReLU(),
                #nn.Conv2d(256, 512, 1, 1),
                #nn.BatchNorm2d(512),
                #nn.ReLU(),
                #nn.Conv2d(512, 512, 3, 1),
                #nn.BatchNorm2d(512),
                #nn.ReLU()
                )


    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x1 = self.maxpool3(x)
        x2 = self.maxpool2(x)
        x3 = self.maxpool1(x)

        diffY = x1.size()[1] - x2.size()[1]
        diffX = x1.size()[2] - x2.size()[2]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        diffY = x1.size()[1] - x3.size()[1]
        diffX = x1.size()[2] - x3.size()[2]

        x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x1, x2, x3 = x1[:,None], x2[:,None], x3[:,None]
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.layers(x)

        return x 

#def feature_transform_regularizer(trans):
#    d = trans.size()[1]
#    batchsize = trans.size()[0]
#    I = torch.eye(d)[None, :, :]
#    if trans.is_cuda:
#        I = I.cuda()
#    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
#    return loss


#if __name__ == '__main__':
    #Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    #sim_data = Variable(torch.rand(32,3,2500))
    #trans = STN3d().cuda()
    #out = trans(data)
    #out = trans(sim_data)
    #print('stn', out.size())
    #print('loss', feature_transform_regularizer(out))

    #sim_data_64d = Variable(torch.rand(32, 64, 2500))
    #trans = STNkd(k=64)
    #out = trans(sim_data_64d)
    #print('stn64d', out.size())
    #print('loss', feature_transform_regularizer(out))

    #pointfeat = PointNetfeat(global_feat=True)
    #out, _, _ = pointfeat(sim_data)
    #print('global feat', out.size())

    #pointfeat = PointNetfeat(global_feat=False)
    #out, _, _ = pointfeat(sim_data)
    #print('point feat', out.size())

    #cls = PointNetCls(k = 5).cuda()
    #out, _, _ = cls(data)
    #out, _, _ = cls(sim_data)
    #print('class', out.size())

    #data = np.load("./test.npy", allow_pickle=True)
    #data = Tensor(data).transpose(1,2)
    #seg = PointNetDenseCls(k = 4).cuda()
    #out, _, _ = seg(sim_data)
    #out, _, _ = seg(data)
    #print('seg', out.size())
