import numpy as np
import zarr
import matplotlib.pyplot as plt
import sys

def visualize_depth_image(zarr_path, frame_idx=0):
    """
    从Zarr文件中加载深度图像并可视化
    
    参数:
        zarr_path (str): Zarr文件的路径
        frame_idx (int): 要可视化的帧索引
    """
    try:
        print(f"正在加载Zarr文件: {zarr_path}")
        zarr_root = zarr.open(zarr_path, mode='r')
        
        print("Zarr文件内容:", list(zarr_root.keys()))
        
        if 'depth' not in zarr_root:
            print("错误：Zarr文件中没有找到'depth'数据集")
            return None  # 返回None表示失败            
        depth = zarr_root['depth']
        print(f"深度数据形状: {depth.shape}, 数据类型: {depth.dtype}")
        
        # 打印shape[0]和shape[1:]
        print(f"总帧数(depth.shape[0]): {depth.shape[0]}")
        print(f"每帧尺寸(depth.shape[1:]): {depth.shape[1:]}")
        
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
        
        return depth  # 返回depth对象以便外部使用
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None

if __name__ == "__main__":
        
    zarr_path = "/home/arxpro/ARX_Remote_Control/data/9_2_lemon_plate.zarr/data"
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    # 调用函数并获取返回的depth对象
    depth = visualize_depth_image(zarr_path, frame_idx)
    
    # 如果成功获取depth对象，可以在这里进一步处理
    if depth is not None:
        print("\n在主程序中获取的深度数据信息:")
        print(f"总帧数: {depth.shape[0]}")
        print(f"每帧尺寸: {depth.shape[1:]}")