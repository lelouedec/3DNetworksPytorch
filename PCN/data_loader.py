import open3d 
import numpy as np
from torch.utils.data.dataset import Dataset
import torch

class Shapenet_dataset(Dataset):
    def __init__(self,liste):
        self.list  = liste

    def __getitem__(self, index):
        pcd_gt    = torch.from_numpy(np.asarray(open3d.read_point_cloud("./shapenet/test/complete/"+self.list[index]+".pcd").points)).float()
        pcd_input = self.resample_pcd(torch.from_numpy(np.asarray(open3d.read_point_cloud("./shapenet/test/partial/"+self.list[index]+".pcd").points)).float(),1024)
        return (pcd_input,pcd_gt)

    def __len__(self):
        return len(self.list)

    def resample_pcd(self,pcd, n):
        """Drop or duplicate points so that pcd has exactly n points"""
        idx = np.random.permutation(pcd.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
        return pcd[idx[:n]]

def load_data(path):
    with open(path+"/test.list") as file:
        model_list = file.read().splitlines()
    return Shapenet_dataset(model_list)

if __name__ == '__main__':
    load_data("shapenet")
    