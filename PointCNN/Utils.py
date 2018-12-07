import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from train_loader import PointCloudDataset
from torch import cuda, FloatTensor, LongTensor
from xConv import XConv,Dense
import DFileParser
import knn
import sys
import open3d

from sklearn.neighbors import LSHForest, NearestNeighbors
#import matplotlib.pyplot as plt
params = {'batch_size': 5,
          'shuffle': False,
          'num_workers': 4}
def knn_indices_func_cpu(rep_pts : FloatTensor,  # (N, pts, dim)
                         pts : FloatTensor,      # (N, x, dim)
                         K : int, D : int
                        ) -> LongTensor:         # (N, pts, K)
    """
    CPU-based Indexing function based on K-Nearest Neighbors search.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: dilatation factor
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    if rep_pts.is_cuda:
        rep_pts = rep_pts.cpu()
    if pts.is_cuda:
        pts = pts.cpu()
    rep_pts = rep_pts.data.numpy()
    pts = pts.data.numpy()

    region_idx = []

    for n, p in enumerate(rep_pts):
        P_particular = pts[n]
        nbrs = NearestNeighbors(D*K + 1, algorithm = "auto").fit(P_particular)
        indices = nbrs.kneighbors(p)[1]
        region_idx.append(indices[:,1::D])

    region_idx = torch.from_numpy(np.stack(region_idx, axis = 0))

    return region_idx


def display_prediction(pc,pred):
    points = np.array(pc).reshape((np.array(pc).shape[0]*5,4000,3)).reshape((np.array(pred).shape[0]*5*4000,3))
    annotations = np.array(pred).reshape((np.array(pred).shape[0]*5,4000,2)).reshape((np.array(pred).shape[0]*5*4000,2))
    colors = []
    for a in annotations:
        if(a[0]>a[1]):
            colors.append(np.array([1.0,0.0,0.0]))
        else:
            colors.append(np.array([0.0,1.0,0.0]))
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(points)
    pcd.colors = open3d.Vector3dVector(np.array(colors))
    open3d.draw_geometries([pcd])


def predict(model_path):
    dataset = PointCloudDataset("./Data/riseholme/Annotated/test")
    training_generator = data.DataLoader(dataset, **params)
    model = torch.load(model_path)
    pc   = []
    pred = []
    count = 0
    for batch in training_generator:
        input = Variable(batch[0]).float()
        #print(input.shape)
        labels = Variable(batch[1]).float()
        #colors = Variable(batch[2].cuda())
        output = model(input)
        _, predicted = torch.max(output.data, 2)
        pc.append(batch[0].data.numpy())
        pred.append(output.data.numpy())
        #print("labels"  + str(labels.shape))
        if(count >= 60):
            display_prediction(pc,pred)
            pc = []
            pred = []
            count = 0

        count = count + 1

def predict_withmodel(model):
    dataset = PointCloudDataset("./Data/riseholme/Annotated/test")
    training_generator = data.DataLoader(dataset, **params)
    pc   = []
    pred = []
    count = 0
    model = model.cpu()
    for batch in training_generator:
        input = Variable(batch[0]).float()
        #print(input.shape)
        labels = Variable(batch[1]).float()
        #colors = Variable(batch[2].cuda())
        output = model(input)
        _, predicted = torch.max(output.data, 2)
        pc.append(batch[0].data.numpy())
        pred.append(output.data.numpy())
        #print("labels"  + str(labels.shape))
        if(count >= 60):
            display_prediction(pc,pred)
            pc = []
            pred = []
            count = 0

        count = count + 1
    #model = model.cuda()
