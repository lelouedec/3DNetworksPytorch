import sys
sys.path.insert(0,'..')
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from time import time
import point
from Utils.net_utils import *


class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2SemSeg, self).__init__()
        self.sa05 = PointNetSetAbstractionMsg(2048,  [0.1, 0.2, 0.4], [32, 64, 128], 6,[[32, 32, 64], [64, 64, 64], [64, 96, 128]], False)
        self.sa1 = PointNetSetAbstraction(1024, 0.2, 32, 128+64+64+3, [32, 32, 64], False)# npoint, radius, nsample, in_channel, mlp, group_all
        self.sa2 = PointNetSetAbstraction(256, 0.4, 32, 64 + 3, [64, 64, 128], False)

        self.fp2 = PointNetFeaturePropagation(192, [512, 256])#in_channel, mlp
        self.fp1 = PointNetFeaturePropagation(512, [256, 128])#in_channel, mlp
        self.fp05 = PointNetFeaturePropagation(131, [128, 64])
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128,num_classes, 1)




    def forward(self, xyz,color):
        xyz = xyz.permute(0, 2, 1)
        color = color.permute(0, 2, 1)
        l05_xyz, l05_points = self.sa05(xyz, color)
        l1_xyz, l1_points = self.sa1(l05_xyz, l05_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)


        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l05_points = self.fp1(l05_xyz, l1_xyz, l05_points, l1_points)
        l0_points = self.fp05(xyz, l05_xyz, color, l05_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)

        return torch.sigmoid(x)## we are adding a sigmoid as it is a binary classification tmtc

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)

if __name__ == '__main__':
    for i in range(10):
        xyz = torch.rand(1, 30000,3).cuda()
        colors = torch.rand(1, 30000,3).cuda()
        net = PointNet2SemSeg(2)
        net.cuda()
        x = net(xyz,colors)
        print(x.shape)
