
import open3d as o3d
import numpy as np
import zarr
from typing import List, Tuple, Optional

class FilterPlane:
# 预定义颜色方案 (R,G,B)
    COLORS = [
        [1, 0, 0],   # 红色 - 第一个平面
        [0, 0, 0],   # 黑色 - 第二个平面
        [1, 1, 1],   # 白色 - 第三个平面
        [1, 0.84, 0] # 金色 - 第四个平面(备用)
    ]
    
    def __init__(self, distance_threshold=0.02, ransac_n=3, max_iterations=1000, min_plane_points=100):
        """
        初始化平面过滤器
        
        参数:
            distance_threshold: 平面内点距离阈值
            ransac_n: RANSAC采样点数
            max_iterations: RANSAC最大迭代次数
            min_plane_points: 被视为有效平面的最小点数
        """
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.max_iterations = max_iterations
        self.min_plane_points = min_plane_points
        self.detected_planes = []  # 保存检测到的平面信息(平面模型和内点索引)
        self.colored_cloud = None  # 保存着色后的点云
    
    def findPlane(self, pcd: o3d.geometry.PointCloud) -> Optional[Tuple[np.ndarray, List[int]]]:
        """
        查找点云中的一个主要平面
        
        返回:
            如果找到平面: 返回(平面模型, 内点索引)
            如果未找到平面: 返回None
        """
        if len(pcd.points) < self.ransac_n:
            return None
            
        # 使用RANSAC检测平面
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.max_iterations
        )
        
        # 检查内点数量是否足够
        if len(inliers) >= self.min_plane_points:
            return (plane_model, inliers)
        return None
    
    def filterPlane(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        过滤掉点云中的所有平面
        
        返回:
            过滤掉所有平面后的点云
        """
        remaining_cloud = pcd
        self.detected_planes.clear()
        
        while True:
            # 检测平面
            result = self.findPlane(remaining_cloud)
            if result is None:
                break
                
            plane_model, inliers = result
            self.detected_planes.append((plane_model, inliers))
            
            # 移除内点
            remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)
            
            # 如果剩余点太少，停止处理
            if len(remaining_cloud.points) < self.ransac_n:
                break
                
        return remaining_cloud
    
    def colorPlanes(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        给点云中的平面上色并返回着色后的点云
        
        返回:
            着色后的点云(平面点着色，非平面点保持原色)
        """
        # 创建点云副本用于着色
        colored_cloud = pcd.clone()
        colors = np.asarray(colored_cloud.colors if colored_cloud.has_colors() else np.ones((len(pcd.points), 3)))
        
        self.detected_planes.clear()
        remaining_indices = set(range(len(pcd.points)))
        plane_count = 0
        
        while True:
            # 从剩余点中检测平面
            remaining_cloud = colored_cloud.select_by_index(list(remaining_indices))
            result = self.findPlane(remaining_cloud)
            
            if result is None:
                break
                
            plane_model, inliers = result
            self.detected_planes.append((plane_model, inliers))
            
            # 转换局部索引回全局索引
            global_inliers = [list(remaining_indices)[i] for i in inliers]
            
            # 应用颜色
            if plane_count < len(self.COLORS):
                color = self.COLORS[plane_count]
            else:
                # 如果平面多于预定义颜色，使用随机颜色
                color = np.random.rand(3).tolist()
                
            colors[global_inliers] = color
            
            # 更新剩余点索引
            remaining_indices -= set(global_inliers)
            plane_count += 1
            
            # 如果剩余点太少，停止处理
            if len(remaining_indices) < self.ransac_n:
                break
                
        # 应用颜色到点云
        colored_cloud.colors = o3d.utility.Vector3dVector(colors)
        self.colored_cloud = colored_cloud
        
        print(f"检测到平面数量: {plane_count}")
        return colored_cloud

    def get_detected_planes(self) -> List[Tuple[np.ndarray, List[int]]]:
        """获取检测到的平面信息"""
        return self.detected_planes

def main():
    """主函数：加载点云数据，去除平面后可视化"""
    
    # ---------------------- 1. 加载数据 ----------------------
    # 定义Zarr文件路径（根据实际路径修改）
    zarr_path = "/home/slam/3D-Diffusion-Policy/3D-Diffusion-Policy/data/5_18_simple_2.zarr/data/processed_point_clouds.zarr"
    frame_idx = 1  # 选择要处理的帧索引
    
    # 从Zarr文件加载点云数据
    zarr_root = zarr.open(zarr_path, mode='r')
    point_clouds = zarr_root['pointcloud']  # 形状 [N, 1014, 6]
    pc_data = point_clouds[frame_idx]       # 获取指定帧数据
    
    # 提取坐标信息（忽略颜色）
    points = pc_data[:, :3]  # 仅取x,y,z坐标

    # 添加数据验证
    print(f"加载数据形状: {points.shape}")  # 应该输出 (1024, 3)
    if points.shape[0] == 0:
        raise ValueError("加载的点云数据为空！检查文件路径和帧索引")
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    print(f"原始点云点数: {len(pcd.points)}")

    # ---------------------- 2. 平面过滤处理 ----------------------
    # 创建平面过滤器实例（参数可根据需要调整）
    filter = FilterPlane(
        distance_threshold=0.001, 
        ransac_n=3,
        max_iterations=50000,
        min_plane_points=100
    )
    
    # 执行平面过滤，得到去除平面后的剩余点云
    remaining_cloud = filter.filterPlane(pcd)
    print(f"过滤后剩余点数: {len(remaining_cloud.points)}")
    
    # 为剩余点云设置颜色（这里设为蓝色）
    remaining_cloud.paint_uniform_color([0, 0, 1])  # RGB值范围[0,1]

    # ---------------------- 3. 可视化配置 ----------------------
    # 创建坐标系（红色-X，绿色-Y，蓝色-Z）
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 添加几何体到可视化器
    vis.add_geometry(remaining_cloud)  # 添加过滤后的点云
    vis.add_geometry(coord_frame)      # 添加坐标系
    
    # 获取渲染选项并配置
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.5, 0.5, 0.5])  # 中灰色背景
    render_opt.point_size = 3.0         # 设置点云显示大小

    # ---------------------- 4. 运行可视化 ----------------------
    print("启动可视化窗口...")
    vis.run()          # 运行可视化循环
    vis.destroy_window()  # 关闭窗口

if __name__ == "__main__":
    main()
