#for not sampling:
import os
import numpy as np
from Convert_PointCloud import PointCloudGenerator
from Cloud_Process import preprocess_point_cloud, farthest_point_sampling
from Inputing_zarr import read_in_depth, generate_pcd_zarr
from FilteringPlane import FilterPlane  # Replace with actual module name
import open3d as o3d
import zarr
from typing import List, Tuple, Optional
import time

# 1. 初始化参数
# 输入路径
zarr_path = "/home/arxpro/ARX_Remote_Control/data/9_3_lemon_plate_2.zarr/data"
# 输出路径：同级新建 pointcloud 文件夹
input_dir = os.path.dirname(zarr_path)
output_dir = os.path.join(os.path.dirname(input_dir), "point_cloud")
os.makedirs(output_dir, exist_ok=True)
output_zarr_path = os.path.join(output_dir, os.path.basename(input_dir))


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
# start, end = 0, depth_from_robot.shape[0]-1 # 想要的范围
start, end = 0, 15999 # 想要的范围
selected_frames = depth_from_robot[start : end+1] 

# 在循环开始前初始化计时变量
total_time = 0
frame_times = []

for i, current_depth in enumerate(selected_frames, start=start): 
    print(f"\n正在处理第 {i} 帧...")
    start_time = time.time()  # 记录帧处理开始时间
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

        processed_points = preprocess_point_cloud(points)  # 确保输入为NumPy数组
        print(f"处理后点云形状: {processed_points.shape}")
        
        
        # 验证最终输出
        if processed_points.shape == (1024, 3):
            all_processed_points.append(processed_points)
            valid_frames += 1
        else:
            print(f"警告: 无效的输出形状 {processed_points.shape}")
            all_processed_points.append(np.zeros((1024, 6)))
            
    except Exception as e:
        print(f"处理第 {i} 帧时出错: {str(e)}")
        all_processed_points.append(np.zeros((1024, 6)))
        # 在每帧处理结束时计算耗时
    frame_time = time.time() - start_time
    frame_times.append(frame_time)
    total_time += frame_time
    print(f"第 {i} 帧处理耗时: {frame_time:.3f} 秒")

# 5. 结果统计和保存
print(f"\n处理完成,有效帧数: {valid_frames}/{depth_from_robot.shape[0]}")

if valid_frames > 0:
    try:
        generate_pcd_zarr(all_processed_points, output_zarr_path)  # 函数内不返回布尔值，直接报错
        print(f"点云已保存至: {output_zarr_path}")
    except Exception as e:
        print(f"保存失败: {str(e)}")


    