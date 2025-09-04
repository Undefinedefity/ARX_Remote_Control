import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
from typing import List
import open3d as o3d

def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


class PointCloudGenerator(object):
    """
    initialization function

    @param min_bound: If not None, list len(3) containing smallest x, y, and z
        values that will not be cropped
    @param max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """
    def __init__(self,  img_size=480):
        self.img_width = img_size
        self.img_height = img_size
        self.cam_mat = np.array([
            [390.64122761, 0, 450.30159532],
            [0, 392.57988967, 200.70517741],
            [0, 0, 1]
          ])   
        extrinsic_matrix = np.array([[ -0.75422983,  -0.23041472, 0.6148548,  -700     ],
            [-0.65643515,  0.28624051, -0.69796795, 800    ],
            [ -0.01517426,  -0.93004055,  -0.36714346, 545  ],
            [ 0.,          0. ,         0. ,         1.,        ]])
        self.extrinsic_matrix = extrinsic_matrix
    
    def generateCroppedPointCloud(self, depth_data, save_img_dir=None, device_id=0):
        #color_img, depth = self.captureImage(cam_name, capture_depth=True, device_id=device_id)

        #if save_img_dir is not None:
         #   self.saveImg(depth, save_img_dir, f"depth_test_{cam_name}")
         #   self.saveImg(color_img, save_img_dir, f"color_test_{cam_name}")
        
        od_cammat = cammat2o3d(self.cam_mat, self.img_width, self.img_height)
        od_depth = o3d.geometry.Image(depth_data)
        o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)
        # 计算相机到世界的变换矩阵
        c2w = self.extrinsic_matrix
        transformed_cloud = o3d_cloud.transform(c2w)

        return np.asarray(transformed_cloud.points), depth_data.squeeze()
        # return np.asarray(o3d_cloud.points), depth_data.squeeze()