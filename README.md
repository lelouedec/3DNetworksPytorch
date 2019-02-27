# 3DNetworksPytorch



## PointSift
An implementation of PointSift using Pytorch (https://arxiv.org/pdf/1807.00652.pdf) lies in the PoinSift folder.
The C_utils folder contains some algorithms inplemented in CUDA and C++ taken from the original implementation of PointSift (https://github.com/MVIG-SJTU/pointSIFT) but wrapped to be used with Pytorch Tensor directly.

## PointCNN
An implementation of PointCNN using Pytorch (https://arxiv.org/pdf/1801.07791.pdf) lies in the PointCNN folder.

## PointNet++
An implementation of PointNet++ using Pytorch (https://arxiv.org/pdf/1706.02413.pdf) lies in the PointNet++ folder.
It uses the same algorithms on GPU as PointSift as Pointsift uses Pointnet++ modules.

All the GPU code work only with Pytorch 0.4.1, as now they do not support C tensor API anymore. I am working on translating them into C++, to move everything to pytorch 1.0
