import numpy as np
import zarr
import matplotlib.pyplot as plt
import sys
import cv2
import os
from tqdm import tqdm
from matplotlib.colors import Normalize

def visualize_depth_image(zarr_path, frame_idx=0):
    """
    从Zarr文件中加载深度图像并可视化
    
    参数:
        zarr_path (str): Zarr文件的路径
        frame_idx (int): 要可视化的帧索引
        
    返回:
        zarr.Array: 深度数据数组对象
    """
    try:
        print(f"正在加载Zarr文件: {zarr_path}")
        zarr_root = zarr.open(zarr_path, mode='r')
        
        print("Zarr文件内容:", list(zarr_root.keys()))
        
        if 'depth' not in zarr_root:
            print("错误：Zarr文件中没有找到'depth'数据集")
            return None
            
        depth = zarr_root['depth']
        print(f"深度数据形状: {depth.shape}, 数据类型: {depth.dtype}")
        print(f"总帧数: {depth.shape[0]}, 每帧尺寸: {depth.shape[1:]}")
        
        if frame_idx < 0 or frame_idx >= len(depth):
            print(f"错误：帧索引{frame_idx}超出范围(0-{len(depth)-1})")
            return None
            
        depth_data = depth[frame_idx]
        print(f"帧 {frame_idx} 的深度数据范围: {np.min(depth_data)} - {np.max(depth_data)}")
        
        plt.figure(figsize=(10, 6))
        plt.imshow(depth_data, cmap='viridis')
        plt.colorbar(label='深度值')
        plt.title(f'深度图像 - 帧 {frame_idx}')
        plt.show()
        
        return depth
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None

def create_depth_video(depth, output_path, fps=30, colormap='viridis'):
    """
    从深度数据创建视频文件
    
    参数:
        depth (zarr.Array): 深度数据数组
        output_path (str): 输出视频文件路径
        fps (int): 视频帧率
        colormap (str): 使用的颜色映射
    单帧：
    python script.py /path/to/data.zarr 10
    生成视频：
    
    """
    try:
        # 获取深度数据的全局最小最大值用于归一化
        print("计算深度数据的全局范围...")
        min_val = np.min(depth)
        max_val = np.max(depth)
        print(f"全局深度范围: {min_val} - {max_val}")
        
        # 创建归一化对象
        norm = Normalize(vmin=min_val, vmax=max_val)
        
        # 获取颜色映射
        cmap = plt.get_cmap(colormap)
        
        # 创建视频写入对象
        height, width = depth.shape[1], depth.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4格式
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"开始生成视频: {output_path}")
        print(f"视频尺寸: {width}x{height}, 帧率: {fps} fps")
        
        # 处理每一帧
        for i in tqdm(range(len(depth)), desc="生成视频帧"):
            frame = depth[i]
            
            # 归一化深度数据并应用颜色映射
            normalized = norm(frame)
            colored = cmap(normalized)
            
            # 转换为0-255的RGB图像
            rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)
            
            # OpenCV使用BGR格式，所以需要转换
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # 写入视频帧
            video.write(bgr_image)
        
        # 释放视频写入对象
        video.release()
        print(f"视频已成功保存至: {output_path}")
        return True
        
    except Exception as e:
        print(f"创建视频时出错: {str(e)}")
        return False

if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) < 2:
        print("用法: python script.py <zarr_path> [frame_index] [output_video]")
        print("示例:")
        print("  显示单帧: python script.py /path/to/data.zarr 10")
        print("  生成视频: python script.py /path/to/data.zarr -1 /path/to/output.mp4")
        sys.exit(1)
    
    zarr_path = sys.argv[1]
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    output_video = sys.argv[3] if len(sys.argv) > 3 else None
    
    # 加载深度数据
    depth = visualize_depth_image(zarr_path, frame_idx)
    
    if depth is not None:
        # 如果指定了输出视频路径，则创建视频
        if output_video:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_video), exist_ok=True)
            
            # 创建视频
            success = create_depth_video(depth, output_video, fps=30)
            
            if success:
                print("视频生成完成!")
            else:
                print("视频生成失败")
        else:
            print("\n深度数据信息:")
            print(f"总帧数: {depth.shape[0]}")
            print(f"每帧尺寸: {depth.shape[1:]}")