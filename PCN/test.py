import open3d
import models 
import data_loader
import torch.optim as optim
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


model = torch.load("./models/model_60000.ckpt").cuda()
dataset = data_loader.load_data("shapenet")
my_dataset_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=1,shuffle=False)
for input_tensor,gt_tensor in my_dataset_loader:
    input_tensor =input_tensor.cuda()
    gt_tensor = gt_tensor.cuda()
    coarse,fine = model(input_tensor)
    print(coarse.shape,fine.shape,gt_tensor.shape)
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(fine.data.cpu().numpy()[0]+np.array([1.0,0.0,0.0]))
    pcd.colors = open3d.Vector3dVector(np.ones((fine.shape[1],3))* [0.76,0.23,0.14])

    pcd2 = open3d.PointCloud()
    pcd2.points = open3d.Vector3dVector(gt_tensor.data.cpu().numpy()[0]+np.array([-1.0,0.0,0.0]))
    pcd2.colors = open3d.Vector3dVector(np.ones((gt_tensor.shape[1],3))* [0.16,0.23,0.14])

    pcd3 = open3d.PointCloud()
    pcd3.points = open3d.Vector3dVector(input_tensor.data.cpu().numpy()[0])
    pcd3.colors = open3d.Vector3dVector(np.ones((input_tensor.shape[1],3))* [0.16,0.23,0.14])
    open3d.draw_geometries([pcd,pcd2,pcd3])
