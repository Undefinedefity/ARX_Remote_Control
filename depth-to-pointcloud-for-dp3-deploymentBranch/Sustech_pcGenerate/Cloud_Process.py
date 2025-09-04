import os
import zarr
import pickle
import tqdm
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
import re
import time


import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
import socket
import pickle


def farthest_point_sampling(points, num_points=1024, use_cuda=True): #For test, increase the number of the sampled points. Original: 1024
    #points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        raise ValueError("输入点云为空数组！")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"输入需为 (N,3) 数组，但得到形状 {points.shape}")
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices

def preprocess_point_cloud(points, use_cuda=True):
    num_points = 1024

    WORK_SPACE = [
    [-0.05, 0.1],
    [-0.04, 0.2],
    [-0.0025, 0.1]
]
    # scale
    points = points[..., :3]*0.0002500000118743628
     # crop
    points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
                                (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
                                (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]
    point_xyz = points[..., :3]
    points_xyz, _= farthest_point_sampling(point_xyz, num_points, use_cuda)
    return points_xyz

def boundary(WORK_SPACE):
    min_bound = np.array([WORK_SPACE[0][0], WORK_SPACE[1][0], WORK_SPACE[2][0]])  # 最小角点 [x_min, y_min, z_min]
    max_bound = np.array([WORK_SPACE[0][1], WORK_SPACE[1][1], WORK_SPACE[2][1]])  # 最大角点 [x_max, y_max, z_max]
    return min_bound, max_bound