import os
import numpy as np
from Convert_PointCloud import PointCloudGenerator
from Cloud_Process import preprocess_point_cloud, farthest_point_sampling
from Inputing_zarr import read_in_depth, generate_pcd_zarr
from FilteringPlane import FilterPlane  # Replace with actual module name
import open3d as o3d
import zarr
from typing import List, Tuple, Optional

# 1. 初始化参数
zarr_path = "/home/slam/3D-Diffusion-Policy/3D-Diffusion-Policy/data/5_26.zarr/data"
output_zarr_path = "/home/slam/3D-Diffusion-Policy/3D-Diffusion-Policy/data/5_26.zarr/data/realpcd"


# 2. 读取深度数据（添加详细检查）
print("正在加载深度数据...")
depth_from_robot = read_in_depth(zarr_path)
print(f"深度数据形状: {depth_from_robot.shape}, 数据类型: {depth_from_robot.dtype}")

# 3. 初始化点云生成器（添加调试信息）
pc_generator = PointCloudGenerator(
    img_size=depth_from_robot.shape[1]  # 使用深度图高度
)
print("点云生成器初始化完成")

# 4. 处理流程优化
all_processed_points = []
valid_frames = 0
# total_frames = min(5, depth_from_robot.shape[0])  # 需要显示总帧数

# for i in range(min(1, depth_from_robot.shape[0])):  # 先只处理前5帧用于调试

# switch a method for looping:
start, end = 80, 80  # 想要的范围
selected_frames = depth_from_robot[start : end+1]  # 切片获取980-984（共5帧）

for i, current_depth in enumerate(selected_frames, start=start): 
    print(f"\n正在处理第 {i} 帧...")
    # current_depth = depth_from_robot[i] #when using enumerate, this line is not needed
    
    # 检查深度图有效性
    print(f"深度图范围: {np.min(current_depth)} - {np.max(current_depth)}")
    if np.all(current_depth == 0):
        print("警告: 深度图全为零值")
        all_processed_points.append(np.zeros((1024, 6)))
        continue
    
    try:
        # 生成点云（添加详细日志）
        print("正在生成点云...")
        points, _ = pc_generator.generateCroppedPointCloud(
            depth_data=current_depth,
        )
        print(f"生成点云形状: {points.shape if isinstance(points, np.ndarray) else '无效'}")
        
        if not isinstance(points, np.ndarray) or points.size == 0:
            raise ValueError("生成的点云为空")
            
        #filter
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        print(f"原始点云点数: {len(pcd.points)}")
        # 创建平面过滤器实例（参数可根据需要调整）
        filter = FilterPlane(
        distance_threshold=0, 
        ransac_n=3,
        max_iterations=1000,
        min_plane_points=80
        )
        # 执行平面过滤，得到去除平面后的剩余点云
        filtered_pcd = filter.filterPlane(pcd)
        print(f"过滤后剩余点数: {len(filtered_pcd.points)}")
        
        # 检查是否为空
        if len(filtered_pcd.points) == 0:
            print("警告: 平面过滤后无剩余点")
            all_processed_points.append(np.zeros((1024, 3)))
            continue

        # 转换为NumPy数组并采样
        filtered_points_np = np.asarray(filtered_pcd.points)
        processed_points = preprocess_point_cloud(filtered_points_np)  # 确保输入为NumPy数组
        print(f"处理后点云形状: {processed_points.shape}")
        all_processed_points.append(processed_points)
        valid_frames += 1

            
    except Exception as e:
        print(f"处理第 {i} 帧时出错: {str(e)}")
        all_processed_points.append(np.zeros((1024, 6)))

# 5. 结果统计和保存
print(f"\n处理完成,有效帧数: {valid_frames}/{depth_from_robot.shape[0]}")

if valid_frames > 0:
    try:
        all_processed_points = np.stack(all_processed_points, axis=0)
        print(f"最终点云数组形状: {all_processed_points.shape}")
        
        # 保存结果
        success = generate_pcd_zarr(all_processed_points, output_zarr_path)
        if success:
            print(f"点云已保存至: {output_zarr_path}")
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")
else:
    print("警告: 没有有效帧被处理")