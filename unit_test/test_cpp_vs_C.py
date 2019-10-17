import sys
sys.path.insert(0,'..')
import open3d as opend
import torch
import numpy as np


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

if(len(sys.argv)>1):
    if(sys.argv[1]=="C"):
        from C_utils import libsift ###C version, use venv using pytorch 0.4
        def group_points(xyz,idx):
            b , n , c = xyz.shape
            m = idx.shape[1]
            nsample = idx.shape[2]
            out = torch.zeros((xyz.shape[0],xyz.shape[1], idx.shape[2],c)).cuda()
            libsift.group_points(b,n,c,n,nsample,xyz,idx.int(),out)
            np.save("grouped_points.npy", out)

        def farthest_point_sample_gpu(xyz, npoint):
            b, n ,c = xyz.shape
            centroid = torch.zeros((xyz.shape[0],npoint), dtype=torch.int32).cuda()
            temp = torch.zeros((32,n)).cuda()
            libsift.farthestPoint(b,n, npoint, xyz , temp   ,centroid)
            np.save("centroids.npy", centroid.long().cpu().numpy() )

        def ball_query(radius, nsample, xyz, new_xyz):
            b, n ,c = xyz.shape
            m =  new_xyz.shape[1]
            group_idx = torch.zeros((new_xyz.shape[0],new_xyz.shape[1], nsample), dtype=torch.int32).cuda()
            pts_cnt = torch.zeros((xyz.shape[0],xyz.shape[1]), dtype=torch.int32).cuda()
            libsift.ball_query (b, n, m, radius, nsample, xyz, new_xyz, group_idx ,pts_cnt)
            np.save("group_idx.npy",group_idx.long().cpu().numpy())

    elif(sys.argv[1]=="Cpp"):
        import point ###Cpp version, use venv using pytorch 1.0+
        def group_points(xyz,idx):
            b , n , c = xyz.shape
            m = idx.shape[1]
            nsample = idx.shape[2]
            out = torch.zeros((xyz.shape[0],xyz.shape[1], idx.shape[2],c)).cuda()
            point.group_points(b,n,c,n,nsample,xyz,idx.int(),out)
            np.save("grouped_points_cpp.npy", out)

        def farthest_point_sample_gpu(xyz, npoint):
            b, n ,c = xyz.shape
            centroid = torch.zeros((xyz.shape[0],npoint), dtype=torch.int32).cuda()
            temp = torch.zeros((32,n)).cuda()
            point.farthestPoint(b,n, npoint, xyz , temp   ,centroid)
            np.save("centroids_cpp.npy", centroid.long().cpu().numpy() )

        def ball_query(radius, nsample, xyz, new_xyz):
            b, n ,c = xyz.shape
            m =  new_xyz.shape[1]
            group_idx = torch.zeros((new_xyz.shape[0],new_xyz.shape[1], nsample), dtype=torch.int32).cuda()
            pts_cnt = torch.zeros((xyz.shape[0],xyz.shape[1]), dtype=torch.int32).cuda()
            point.ball_query (b, n, m, radius, nsample, xyz, new_xyz, group_idx ,pts_cnt)
            np.save("group_idxcpp.npy",group_idx.long().cpu().numpy())
else:
    print("Please select the extension you want to use !! ")

def test_farthest():
    pc = torch.from_numpy(np.asarray(opend.read_point_cloud("test_pc.ply").points)).unsqueeze(0).float().cuda()
    farthest_point_sample_gpu(pc,500)
def test_ball_query():
    pc = torch.from_numpy(np.asarray(opend.read_point_cloud("test_pc.ply").points)).unsqueeze(0).float().cuda()
    centroids = torch.from_numpy(np.load('centroids.npy')).cuda()
    new_xyz =  index_points(pc, centroids)
    ball_query(0.1, 32, pc, new_xyz)

def test_group_points():
    pc = torch.from_numpy(np.asarray(opend.read_point_cloud("test_pc.ply").points)).unsqueeze(0).float().cuda()
    centroids = torch.from_numpy(np.load('centroids.npy')).cuda()
# test_farthest()
# test_ball_query()
def test_arrays():
    cpp_f = np.load('centroids.npy')
    c_f = np.load('centroids_cpp.npy')
    if(cpp_f.all()==c_f.all()):
        print("gotcha centroids")
    cpp_g = np.load('group_idx.npy')
    c_g = np.load('group_idxcpp.npy')
    if(cpp_g.all()== c_g.all()):
        print("gotcha groups")
test_arrays()
