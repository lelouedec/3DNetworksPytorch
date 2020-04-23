import math
from torch import nn
from torch.autograd import Function
import torch
import sys
import time
import point

x = torch.FloatTensor(1,200, 3).cuda()
ya = torch.zeros((1,200, 8), dtype=torch.int32).cuda()
radius = 1.0
start_time = time.time()
point.select_cube(x,ya,8,6,radius)
print(time.time() - start_time)
ya = ya.cpu()

start_time = time.time()
xyz = x.cpu()
radius = 0.4
Dist = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
B, N, _ = xyz.shape
idx = torch.empty(B, N, 8)
judge_dist = radius ** 2
temp_dist = torch.ones(B, N, 8) * 1e10
for b in range(B):
    for n in range(N):
        idx[b, n, :] = n
        x, y, z = xyz[b, n]
        for p in range(N):
            if p == n: continue
            tx, ty, tz = xyz[b, p]
            dist = Dist(x - tx, y - ty, z - tz)
            if dist > judge_dist: continue
            _x, _y, _z = tx > x, ty > y, tz > z
            temp_idx = (_x * 4 + _y * 2 + _z).int()
            if dist < temp_dist[b, n, temp_idx]:
                idx[b, n, temp_idx] = p
                temp_dist[b, n, temp_idx] = dist
print(idx)
print(ya)
if(torch.all(torch.eq(idx.int(), ya.cpu().int()))):
    print("success")

print(time.time() - start_time)
