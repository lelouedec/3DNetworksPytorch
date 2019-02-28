import sys
sys.path.insert(0,'..')
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from time import time
from C_utils import libsift


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, D1, D2, ..., Dn]
    Return:
        new_points:, indexed points data, [B, D1, D2, ..., Dn, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def square_distance(src, dst):
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

def group_points(xyz,idx):
    b , n , c = xyz.shape
    m = idx.shape[1]
    nsample = idx.shape[2]
    out = torch.zeros((xyz.shape[0],xyz.shape[1], idx.shape[2],c)).cuda()
    libsift.group_points(b,n,c,n,nsample,xyz,idx.int(),out)
    return out

def farthest_point_sample_gpu(xyz, npoint):
    b, n ,c = xyz.shape
    centroid = torch.zeros((xyz.shape[0],npoint), dtype=torch.int32).cuda()
    temp = torch.zeros((32,n)).cuda()
    libsift.farthestPoint(b,n, npoint, xyz , temp   ,centroid)
    return centroid.long()

def ball_query(radius, nsample, xyz, new_xyz):
    b, n ,c = xyz.shape
    m =  new_xyz.shape[1]
    group_idx = torch.zeros((new_xyz.shape[0],new_xyz.shape[1], nsample), dtype=torch.int32).cuda()
    pts_cnt = torch.zeros((xyz.shape[0],xyz.shape[1]), dtype=torch.int32).cuda()
    libsift.ball_query (b, n, m, radius, nsample, xyz, new_xyz, group_idx ,pts_cnt)

    return group_idx.long()

def idx_pts(points,idx):
    new_points = torch.cat([points.index_select(1,idx[b]) for b in range(0,idx.shape[0])], dim=0)
    return  new_points

def sample_and_group(npoint, radius, nsample, xyz, points):
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

    new_xyz = index_points(xyz, farthest_point_sample_gpu(xyz, npoint)) # [B,n,3] and [B,np] → [B,np,3]
    idx = ball_query(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)# [B,n,3] and [B,n,M] → [B,n,M,3]
    grouped_xyz -= new_xyz.view(B, Np, 1, C)  # the points of each group will be normalized with their centroid
    if points is not None:
        grouped_points = index_points(points, idx)# [B,n,3] and [B,n,M] → [B,n,M,3]
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def sample_and_group_all(xyz, points):
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


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1,1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all


    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:,:,:3], idx[:,:,:3] #[B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists #[B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1) #[B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim = 2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2SemSeg, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        return x

if __name__ == '__main__':
    for i in range(10):
        xyz = torch.rand(16, 3, 2048).cuda()
        net = PointNet2SemSeg(2048)
        net.cuda()
        x = net(xyz)

    xyz1 = torch.rand(4, 3, 2048).cuda()
    xyz2 = torch.rand(4, 3, 512).cuda()
    points2 = torch.rand(4, 64, 2048).cuda()
    net = PointNetFeaturePropagation(64, [128, 256])
    net.cuda()
    new_points = net(xyz1, xyz2, None, points2)
    print(new_points.shape)

    xyz = torch.rand(8, 3, 2048).cuda()
    net = PointNet2SemSeg(2048)
    net.cuda()
    x = net(xyz)
    print(x.shape)
