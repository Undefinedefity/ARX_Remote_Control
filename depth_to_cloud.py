import zarr
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import copy

# 步骤1：读取zarr文件中的depth数组
def read_zarr_depth(zarr_path):
    # 使用zarr打开文件并读取depth数组
    root = zarr.open(zarr_path, mode='r')
    data = root['data']
    depth_array = data['depth'][:]  # 假设depth数组位于根组下
    return depth_array

# 步骤2：将深度图转换为点云
def depth_to_pointcloud(depth_array, fx, fy, cx, cy):
    episode_idx = 0
    point_cloud_array = []
    num, height, width = depth_array.shape
    while episode_idx < 2:
        idx = 0
        point_cloud_sub_array = []

        for _,depth_map in enumerate(depth_array):
            print(f"Depth to Pointcloud: ep {episode_idx+1} and step {idx}")
            # 生成像素坐标网格
            v, u = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
            # 转换为浮点型并调整单位（假设深度单位为毫米，转为米）
            depth = depth_map.astype(np.float32) / 1000.0
            # 计算点云坐标
            Z = depth
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            point_cloud = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
            print("Sampling")
            point_cloud = point_cloud_sampling(point_cloud,1024,'fps')
            # 合并坐标并展平
            point_cloud_sub_array.append(point_cloud)
            idx = idx + 1
            if idx >= 200:
                episode_idx = episode_idx + 1
                break

        point_cloud_array.extend(copy.deepcopy(point_cloud_sub_array))
    return np.stack(point_cloud_array, axis=0)

def point_cloud_sampling(point_cloud:np.ndarray, num_points:int, method:str='fps'):
    """
    support different point cloud sampling methods
    point_cloud: (N, 6), xyz+rgb or (N, 3), xyz
    """
    if num_points == 'all': # use all points
        return point_cloud
    
    if point_cloud.shape[0] <= num_points:
        # cprint(f"warning: point cloud has {point_cloud.shape[0]} points, but we want to sample {num_points} points", 'yellow')
        # pad with zeros
        point_cloud_dim = point_cloud.shape[-1]
        point_cloud = np.concatenate([point_cloud, np.zeros((num_points - point_cloud.shape[0], point_cloud_dim))], axis=0)
        return point_cloud

    if method == 'uniform':
        # uniform sampling
        sampled_indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        point_cloud = point_cloud[sampled_indices]
    elif method == 'fps':
        # fast point cloud sampling using torch3d
        point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
        num_points = torch.tensor([num_points]).cuda()
        # remember to only use coord to sample
        _, sampled_indices = torch3d_ops.sample_farthest_points(points=point_cloud[...,:3], K=num_points)
        point_cloud = point_cloud.squeeze(0).cpu().numpy()
        point_cloud = point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
    else:
        raise NotImplementedError(f"point cloud sampling method {method} not implemented")

    return point_cloud
    

# 步骤3：存储点云为zarr文件
def save_pointcloud_zarr(point_cloud, output_path):
    print("Begin to save data as zarr")
    root = zarr.open(output_path, mode='w')
    # 创建数据集，适当分块以优化存储
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    point_cloud_chunk_size = (50, point_cloud.shape[1], point_cloud.shape[2])
    root.create_dataset('point_cloud', data=point_cloud, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

# 示例参数（根据实际相机参数修改）
fx = 386.0   # 焦距x
fy = 386.0   # 焦距y
cx = 323.7   # 光心x
cy = 238.3   # 光心y

# 主流程
if __name__ == "__main__":
    # 读取深度图
    depth_array = read_zarr_depth('/home/slam327/ARX_Remote_Control/data/5_18_simple.zarr')  # 输入文件路径
    # 转换为点云
    point_cloud = depth_to_pointcloud(depth_array, fx, fy, cx, cy)
    # 保存结果
    save_pointcloud_zarr(point_cloud, '/home/slam327/ARX_Remote_Control/data/5_18_simple_pointcloud.zarr')  # 输出文件路径