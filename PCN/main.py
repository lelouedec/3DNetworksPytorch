import open3d
import models 
import data_loader
import torch.optim as optim
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
imprt tqdm

model = models.PCNEMD().cuda()
# model = torch.load("./models/model_10000.ckpt").cuda()
alpha = [ 0.01,0.1,0.5,1.0]
lr = 1e-6
optimizer = optim.Adam([{'params': model.parameters(), 'lr': lr}])
dataset = data_loader.load_data("shapenet")
my_dataset_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=1,shuffle=False)
epochs = 70000
writer = SummaryWriter()
a = 1
for p in tqdm.tqdm(range(0,epochs)):
    lost = []
    for input_tensor,gt_tensor in my_dataset_loader:
        optimizer.zero_grad
        input_tensor =input_tensor.cuda()
        gt_tensor = gt_tensor.cuda()
        coarse,fine = model(input_tensor)
        loss =  model.create_loss(coarse,fine,gt_tensor,alpha[a])      
        loss.backward()
        lost.append(loss.data.item())
        optimizer.step()

    if(p%10000==0 and p!=0):
        torch.save(model, "./models/model_"+str(p)+".ckpt")
        model.cuda()
        lr = lr/10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if(p%10==0):
         writer.add_scalar('Loss',np.array(lost).mean(), p)
    if(p==10000):
        a = a + 1
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
    if(p==20000):
        a =  a + 1
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
    if(p==50000):
        a =  a + 1
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
        
