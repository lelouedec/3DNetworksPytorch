import sys
sys.path.insert(0,'..')
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from C_utils import libsift
import time
import open3d
import DFileParser
import torch.optim as optim



def conv_bn(inp, oup, kernel, stride=1, activation='relu'):
    seq = nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride),
        nn.BatchNorm2d(oup)
    )
    if activation == 'relu':
        seq.add_module('2', nn.ReLU())
    return seq

def conv1d_bn(inp, oup, kernel, stride=1, activation='relu'):
    seq = nn.Sequential(
        nn.Conv1d(inp, oup, kernel, stride),
        nn.BatchNorm1d(oup)
    )
    if activation == 'relu':
        seq.add_module('2', nn.ReLU())
    return seq


def fc_bn(inp, oup):
    return nn.Sequential(
        nn.Linear(inp, oup),
        nn.BatchNorm1d(oup),
        nn.ReLU()
)

class PointNet_SA_module_basic(nn.Module):
    def __init__(self):
        super(PointNet_SA_module_basic, self).__init__()

    def index_points(self, points, idx):
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        # new_points = torch.cat([points.index_select(1,idx[b]) for b in range(0,idx.shape[0])], dim=0)
        return  new_points

    def square_distance(self, src, dst):
        """
        Description:
            just the simple Euclidean distance fomula，(x-y)^2,
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1).contiguous())
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def group_points(self,xyz,idx):
        b , n , c = xyz.shape
        m = idx.shape[1]
        nsample = idx.shape[2]
        out = torch.zeros((xyz.shape[0],xyz.shape[1], idx.shape[2],c)).cuda()
        libsift.group_points(b,n,c,n,nsample,xyz,idx.int(),out)
        return out

    def farthest_point_sample_gpu(self, xyz, npoint):
        b, n ,c = xyz.shape
        centroid = torch.zeros((xyz.shape[0],npoint), dtype=torch.int32).cuda()
        temp = torch.zeros((32,n)).cuda()
        libsift.farthestPoint(b,n, npoint, xyz , temp   ,centroid)
        return centroid.long()

    def ball_query(self, radius, nsample, xyz, new_xyz):
        b, n ,c = xyz.shape
        m =  new_xyz.shape[1]
        group_idx = torch.zeros((new_xyz.shape[0],new_xyz.shape[1], nsample), dtype=torch.int32).cuda()
        pts_cnt = torch.zeros((xyz.shape[0],xyz.shape[1]), dtype=torch.int32).cuda()
        libsift.ball_query (b, n, m, radius, nsample, xyz, new_xyz, group_idx ,pts_cnt)

        return group_idx.long()

    def idx_pts(self,points,idx):
        new_points = torch.cat([points.index_select(1,idx[b]) for b in range(0,idx.shape[0])], dim=0)
        return  new_points
    def sample_and_group(self, npoint, radius, nsample, xyz, points):
        """
        Input:
            npoint: the number of points that make the local region.
            radius: the radius of the local region
            nsample: the number of points in a local region
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, 1, C]
            new_points: sampled points data, [B, 1, N, C+D]
        """
        B, N, C = xyz.shape
        Np = npoint
        assert isinstance(Np, int)

        new_xyz = self.idx_pts(xyz, self.farthest_point_sample_gpu(xyz, npoint)) # [B,n,3] and [B,np] → [B,np,3]
        idx = self.ball_query(radius, nsample, xyz, new_xyz)
        grouped_xyz = self.index_points(xyz, idx)# [B,n,3] and [B,n,M] → [B,n,M,3]
        grouped_xyz -= new_xyz.view(B, Np, 1, C)  # the points of each group will be normalized with their centroid
        if points is not None:
            grouped_points = self.index_points(points, idx)# [B,n,3] and [B,n,M] → [B,n,M,3]
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz
        return new_xyz, new_points

    def sample_and_group_all(self, xyz, points):
        """
        Description:
            Equivalent to sample_and_group with npoint=1, radius=np.inf, and the centroid is (0, 0, 0)
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, 1, C]
            new_points: sampled points data, [B, 1, N, C+D]
        """
        device = xyz.device
        B, N, C = xyz.shape
        new_xyz = torch.zeros(B, 1, C).to(device)
        grouped_xyz = xyz.view(B, 1, N, C)
        if points is not None:
            new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        else:
            new_points = grouped_xyz
        return new_xyz, new_points


class Pointnet_SA_module(PointNet_SA_module_basic):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):

        super(Pointnet_SA_module, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.conv_bns = nn.Sequential()
        in_channel += 3  # +3是因为points 与 xyz concat的原因
        for i, out_channel in enumerate(mlp):
            m = conv_bn(in_channel, out_channel, 1)
            self.conv_bns.add_module(str(i), m)
            in_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: the shape is [B, N, 3]
            points: thes shape is [B, N, D], the data include the feature infomation
        Return:
            new_xyz: the shape is [B, Np, 3]
            new_points: the shape is [B, Np, D']
        """

        if self.group_all:
            new_xyz, new_points = self.sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self.sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 1, 2).contiguous()  # change size to (B, C, Np, Ns), adaptive to conv
        # print("1:", new_points.shape)
        new_points = self.conv_bns(new_points)
        #print("2:", new_points.shape)
        new_points = torch.max(new_points, 3)[0]  # 取一个local region里所有sampled point特征对应位置的最大值。

        new_points = new_points.permute(0, 2, 1).contiguous()
        #print(new_points.shape)
        return new_xyz, new_points

class PointSIFT_module_basic(nn.Module):
    def __init__(self):
        super(PointSIFT_module_basic, self).__init__()


    def group_points(self,xyz,idx):
        b , n , c = xyz.shape
        m = idx.shape[1]
        nsample = idx.shape[2]
        out = torch.zeros((xyz.shape[0],xyz.shape[1], 8,c)).cuda()
        libsift.group_points(b,n,c,m,nsample,xyz,idx.int(),out)
        return out

    def pointsift_select(self, radius, xyz):
        y = torch.zeros((xyz.shape[0],xyz.shape[1], 8), dtype=torch.int32).cuda()
        libsift.select_cube(xyz,y,xyz.shape[0],xyz.shape[1],radius)
        return y.long()

    def pointsift_group(self, radius, xyz, points, use_xyz=True):

        B, N, C = xyz.shape
        assert C == 3
        # start_time = time.time()
        idx = self.pointsift_select(radius, xyz)  # B, N, 8
        # print("select SIR 1 ", time.time() - start_time, xyz.shape)

        # start_time = time.time()
        grouped_xyz = self.group_points(xyz, idx)  # B, N, 8, 3
        # print("group SIR SIR 1 ", time.time() - start_time)

        grouped_xyz -= xyz.view(B, N, 1, 3)
        if points is not None:
            grouped_points = self.group_points(points, idx)
            if use_xyz:
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz
        return grouped_xyz, grouped_points, idx

    def pointsift_group_with_idx(self, idx, xyz, points, use_xyz=True):

        B, N, C = xyz.shape
        grouped_xyz = self.group_points(xyz, idx)  # B, N, 8, 3
        grouped_xyz -= xyz.view(B, N, 1, 3)
        if points is not None:
            grouped_points = self.group_points(points, idx)
            if use_xyz:
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz
        return grouped_xyz, grouped_points

class PointSIFT_res_module(PointSIFT_module_basic):

    def __init__(self, radius, output_channel, extra_input_channel=0, merge='add', same_dim=False):
        super(PointSIFT_res_module, self).__init__()
        self.radius = radius
        self.merge = merge
        self.same_dim = same_dim

        self.conv1 = nn.Sequential(
            conv_bn(3 + extra_input_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2])
        )

        self.conv2 = nn.Sequential(
            conv_bn(3 + output_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2], activation=None)
        )
        if same_dim:
            self.convt = nn.Sequential(
                nn.Conv1d(extra_input_channel, output_channel, 1),
                nn.BatchNorm1d(output_channel),
                nn.ReLU()
            )

    def forward(self, xyz, points):
        _, grouped_points, idx = self.pointsift_group(self.radius, xyz, points)  # [B, N, 8, 3], [B, N, 8, 3 + C]

        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()  # B, C, N, 8
        ##print(grouped_points.shape)
        new_points = self.conv1(grouped_points)
        ##print(new_points.shape)
        new_points = new_points.squeeze(-1).permute(0, 2, 1).contiguous()

        _, grouped_points = self.pointsift_group_with_idx(idx, xyz, new_points)
        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()

        ##print(grouped_points.shape)
        new_points = self.conv2(grouped_points)

        new_points = new_points.squeeze(-1)

        if points is not None:
            points = points.permute(0, 2, 1).contiguous()
            # print(points.shape)
            if self.same_dim:
                points = self.convt(points)
            if self.merge == 'add':
                new_points = new_points + points
            elif self.merge == 'concat':
                new_points = torch.cat([new_points, points], dim=1)

        new_points = F.relu(new_points)
        new_points = new_points.permute(0, 2, 1).contiguous()

        return xyz, new_points

class PointSIFT_module(PointSIFT_module_basic):

    def __init__(self, radius, output_channel, extra_input_channel=0, merge='add', same_dim=False):
        super(PointSIFT_module, self).__init__()
        self.radius = radius
        self.merge = merge
        self.same_dim = same_dim

        self.conv1 = nn.Sequential(
            conv_bn(3+extra_input_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2])
        )

        self.conv2 = conv_bn(output_channel, output_channel, [1, 1], [1, 1])


    def forward(self, xyz, points):
        _, grouped_points, idx = self.pointsift_group(self.radius, xyz, points)  # [B, N, 8, 3], [B, N, 8, 3 + C]

        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()  # B, C, N, 8
        ##print(grouped_points.shape)
        new_points = self.conv1(grouped_points)
        new_points = self.conv2(new_points)

        new_points = new_points.squeeze(-1)

        return xyz, new_points

class Pointnet_fp_module(nn.Module):
    def __init__(self,mlp,dimin):
        super(Pointnet_fp_module, self).__init__()
        self.tanh = nn.Hardtanh(min_val=0, max_val=1e-10)
        self.convs = []
        for i,m in enumerate(mlp):
            self.convs.append(conv_bn(dimin[i],m,[1,1],[1,1]).cuda())

    def forward(self,xyz1, xyz2, points1, points2):
        b,n, c  = xyz1.shape
        m = xyz2.shape[1]
        dist = torch.zeros((xyz1.shape[0],xyz1.shape[1], 3)).cpu()
        idx = torch.zeros((xyz1.shape[0],xyz1.shape[1], 3), dtype=torch.int32).cpu()
        libsift.interpolate(b,n, m, xyz1.cpu(), xyz2.cpu(), dist,idx);
        dist     = self.tanh(dist)
        norm     = torch.sum((1.0/dist),dim = 2, keepdim=True)
        norm     = norm.repeat([1,1,3])
        weight   = (1.0/dist) / norm
        interpolated_points = torch.zeros((b,n, points1.shape[2])).cpu()
        libsift.three_interpolate(b, m, c, n, points2.cpu(), idx.cpu(), weight.cpu(), interpolated_points)
        xyz1 = xyz1.cuda()
        xyz2 = xyz2.cuda()
        points1 = points1.cuda()
        points2 = points2.cuda()
        interpolated_points = torch.cat([interpolated_points.cuda(),points1],dim=2)
        interpolated_points = interpolated_points.unsqueeze(2).permute(0,3,2,1)
        for c in range(0,len(self.convs)):
            interpolated_points = self.convs[c](interpolated_points)

        interpolated_points = interpolated_points.squeeze(2)
        return interpolated_points

class PointSIFT(nn.Module):
    def __init__(self,nb_classes):
        super(PointSIFT, self).__init__()

        self.num_classes = nb_classes

        self.pointsift_res_m3 = PointSIFT_res_module(radius=0.1, output_channel=64,  merge='concat')#extra_input_channel=64)
        self.pointnet_sa_m3 = Pointnet_SA_module(npoint=1024, radius=0.1, nsample=32, in_channel=64, mlp=[64, 128],group_all=False)

        self.pointsift_res_m4 = PointSIFT_res_module(radius=0.2, output_channel=128, extra_input_channel=128)
        self.pointnet_sa_m4 = Pointnet_SA_module(npoint=256, radius=0.2, nsample=32, in_channel=128, mlp=[128, 256],group_all=False)

        self.pointsift_res_m5_1 = PointSIFT_res_module(radius=0.2, output_channel=256, extra_input_channel=256)
        self.pointsift_res_m5_2 = PointSIFT_res_module(radius=0.2, output_channel=512, extra_input_channel=256,same_dim=True)

        self.conv1 = conv1d_bn(768, 512, 1, stride=1, activation='none')

        self.pointnet_sa_m6 = Pointnet_SA_module(npoint=64, radius=0.2, nsample=32, in_channel=512, mlp=[512,512],group_all=False)
        self.pointnet_fp_m0 = Pointnet_fp_module([512,512],[512,512])

        self.pointsift_m0   =  PointSIFT_module(radius=0.5, output_channel=512,extra_input_channel=512)

        self.pointsift_m1   =  PointSIFT_module(radius=0.5, output_channel=512,extra_input_channel=512)

        self.pointsift_m2   =  PointSIFT_module(radius=0.5, output_channel=512,extra_input_channel=512)

        self.conv2 = conv1d_bn(512, 512, 1, stride=1, activation='none')

        self.pointnet_fp_m1 = Pointnet_fp_module([256,256],[256,256])

        self.pointsift_m3   =  PointSIFT_module(radius=0.25, output_channel=256,extra_input_channel=256)

        self.pointsift_m4   =  PointSIFT_module(radius=0.25, output_channel=256,extra_input_channel=256)

        self.conv3 = conv1d_bn(256, 256, 1, stride=1, activation='none')

        self.pointnet_fp_m2 = Pointnet_fp_module([128,128,128],[128,128,128])

        self.pointsift_m5  =  PointSIFT_module(radius=0.1, output_channel=128,extra_input_channel=128)

        ### fc

        self.conv_fc  =  conv1d_bn(128, 128, 1, stride=1, activation='none')

        self.drop_fc  =  nn.Dropout(p=0.5)

        self.conv2_fc =  conv1d_bn(128, 2, 1, stride=1, activation='none')




    def forward(self, xyz, points=None):
        """
        Input:
            xyz: is the raw point cloud(B * N * 3)
        Return:
        """
        B = xyz.size()[0]

        l3_xyz, l3_points = self.pointsift_res_m3(xyz, points)
        # print(l3_xyz.shape, l3_points.shape)
        c3_xyz, c3_points = self.pointnet_sa_m3(l3_xyz, l3_points)

        l4_xyz, l4_points = self.pointsift_res_m4(c3_xyz, c3_points)
        c4_xyz, c4_points = self.pointnet_sa_m4(l4_xyz, l4_points)

        l5_xyz, l5_points = self.pointsift_res_m5_1(c4_xyz, c4_points)
        l5_2_xyz, l5_2_points = self.pointsift_res_m5_2(l5_xyz, l5_points)

        l2_cat_points = torch.cat([l5_points, l5_2_points], dim=2)
        fc_l2_points = self.conv1(l2_cat_points.permute(0,2,1)).permute(0,2,1)
        l3b_xyz, l3b_points = self.pointnet_sa_m6(l5_2_xyz,fc_l2_points)
        l2_points = self.pointnet_fp_m0(c4_xyz,l3b_xyz, c4_points,l3_points ).permute(0,2,1)
        _, l2_points_1 = self.pointsift_m0(c4_xyz,l2_points)
        _, l2_points_2 = self.pointsift_m1(c4_xyz,l2_points)
        _, l2_points_3 = self.pointsift_m2(c4_xyz,l2_points)

        l2_points = torch.cat([l2_points_1,l2_points_2,l2_points_3],dim=-1)
        l2_points = self.conv2(l2_points)
        l1_points = self.pointnet_fp_m1(c3_xyz,c4_xyz, c3_points,l2_points ).permute(0,2,1)
        _, l1_points_1 = self.pointsift_m3(c3_xyz,l1_points)
        _, l1_points_2 = self.pointsift_m4(c3_xyz,l1_points)

        l1_points = torch.cat([l1_points_1,l1_points_2], dim =-1)
        l0_points = self.conv3(l1_points)
        l0_points = self.pointnet_fp_m2(l3_xyz,c3_xyz, l3_points,l0_points ).permute(0,2,1)
        _, l0_points_1 = self.pointsift_m5(l3_xyz,l0_points)

        net = self.conv_fc(l0_points_1)
        net = self.drop_fc(net)
        net = self.conv2_fc(net)

        print(net.shape)
        return net




    @staticmethod
    def get_loss(input, target):
        classify_loss = nn.CrossEntropyLoss()
        loss = classify_loss(input, target)
        return loss

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == "__main__":
    model = PointSIFT(2).cuda()
    PC  =  DFileParser.OpenOBJFile("Strawberry.obj")+6.0
    PC2 =  DFileParser.OpenOBJFile("secondobject.obj")
    annotation1 = np.zeros(PC.shape[0])
    annotation2 = np.ones (PC2.shape[0])
    PC  = np.concatenate((PC,PC2),axis=0)
    PC = torch.from_numpy(PC.astype(np.float32)).unsqueeze(0).cuda()
    Annotation = torch.from_numpy(np.concatenate((annotation1,annotation2),axis=0)).unsqueeze(0).cuda()
    print(PC.shape,Annotation.shape)
    optimizer = optim.SGD([{'params' : model.parameters(),'lr' : 0.01}])
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1.0,1.0]).cuda())

    colors = []
    for c in Annotation[0]:
        if(c == 0):
            colors.append([1.0,0.0,0.0])
        else:
            colors.append([0.0,1.0,0.0])
    colors = np.array(colors)
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(PC.cpu().numpy()[0])
    pcd.colors = open3d.Vector3dVector(np.array(colors))
    open3d.draw_geometries([pcd])

    for i in range(0,300):
        optimizer.zero_grad
        out = model(PC)
        loss = criterion(out,Annotation.long())
        print("Loss : ", loss)
        loss.backward()
        optimizer.step()

    colors = []
    out = out.permute(0,2,1)
    for c in out[0]:
        if(torch.max(c.unsqueeze(0),dim=1)[1] ==0):
            colors.append([0.0,1.0,0.0])
        else:
            colors.append([1.0,0.0,0.0])
    colors = np.array(colors)
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(PC.cpu().numpy()[0])
    pcd.colors = open3d.Vector3dVector(np.array(colors))
    open3d.draw_geometries([pcd])
