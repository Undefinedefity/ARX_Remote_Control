import open3d as o3d
import numpy as np
import zarr
import os
import cv2
from tqdm import tqdm
from Cloud_Process import boundary

def generate_pointcloud_video(zarr_path, output_video, frame_range=None, fps=30):
    """
    从Zarr点云数据生成带有边界框和坐标系的视频
    
    参数:
        zarr_path (str): Zarr文件路径
        output_video (str): 输出视频文件路径
        frame_range (tuple): (起始帧, 结束帧) 可选
        fps (int): 视频帧率
    """
    try:
        print(f"正在加载Zarr文件: {zarr_path}")
        zarr_root = zarr.open(zarr_path, mode='r')
        
        if 'pointcloud' not in zarr_root:
            print("错误：Zarr文件中没有找到'pointcloud'数据集")
            return False
            
        point_clouds = zarr_root['pointcloud']
        print(f"点云数据形状: {point_clouds.shape}, 数据类型: {point_clouds.dtype}")
        
        total_frames = point_clouds.shape[0]
        print(f"总帧数: {total_frames}")
        
        # 设置帧范围
        start_frame = frame_range[0] if frame_range else 0
        end_frame = frame_range[1] if frame_range else total_frames - 1
        
        if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
            print(f"错误：无效的帧范围({start_frame}-{end_frame})，有效范围: 0-{total_frames-1}")
            return False
            
        num_frames = end_frame - start_frame + 1
        print(f"将处理帧范围: {start_frame}-{end_frame} ({num_frames}帧)")
        
        # 创建可视化器（离屏渲染）
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)  # 设置为不可见以提高性能
        
        # 添加坐标系（红色-X，绿色-Y，蓝色-Z）
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        vis.add_geometry(coord_frame)
        
        # 添加边界框
        WORK_SPACE = [
            [-0.08, 0.1],
            [-0.1, 0.2],
            [-0.001, 0.1]
        ]
        min_bound, max_bound = boundary(WORK_SPACE)
        custom_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        custom_bbox.color = (0, 1, 0)  # 设置为绿色
        vis.add_geometry(custom_bbox)
        
        # 创建点云对象（初始为空）
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)
        
        # 设置渲染选项
        render_opt = vis.get_render_option()
        render_opt.background_color = np.array([0.5, 0.5, 0.5])  # 中灰色背景
        render_opt.point_size = 3.0
        
        # 获取窗口尺寸的正确方式
        width = 1280  # 设置默认宽度
        height = 720  # 设置默认高度
        
        # 创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4格式
        video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        print(f"开始生成视频: {output_video}")
        print(f"视频尺寸: {width}x{height}, 帧率: {fps} fps")
        
        # 处理每一帧
        for frame_idx in tqdm(range(start_frame, end_frame + 1), desc="生成点云视频"):
            # 在循环中添加检查

            # 获取当前帧的点云数据
            pc_data = point_clouds[frame_idx]
            
            # 仅提取坐标（忽略颜色信息）
            points = pc_data[:, :3]  # 只取xyz坐标
            if len(points) == 0:
                print(f"警告：第 {frame_idx} 帧点云为空")
                continue            
            
            # 更新点云
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 更新几何体
            vis.update_geometry(pcd)
            vis.update_geometry(coord_frame)
            vis.update_geometry(custom_bbox)
            
            # 更新渲染器
            vis.poll_events()
            vis.update_renderer()
            
            # 捕获当前帧图像
            image = vis.capture_screen_float_buffer(do_render=True)
            image = np.asarray(image)
            
            # 转换为0-255的RGB图像
            rgb_image = (image * 255).astype(np.uint8)
            
            # OpenCV使用BGR格式，所以需要转换
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # 写入视频帧
            video.write(bgr_image)
        
        # 释放资源
        video.release()
        vis.destroy_window()
        
        print(f"视频已成功保存至: {output_video}")
        return True
        
    except Exception as e:
        print(f"生成点云视频时出错: {str(e)}")
        return False

if __name__ == "__main__":
    # 示例用法
    zarr_path = "/home/arxpro/ARX_Remote_Control/data/9_2_lemon_plate_2.zarr/data/pcd"
    output_video = "/home/arxpro/ARX_Remote_Control/data/9_2_lemon_plate_2.zarr/data/pointcloud_video.mp4"
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    # 生成视频（使用所有帧）
    success = generate_pointcloud_video(zarr_path, output_video)
    
    if success:
        print("点云视频生成完成!")
    else:
        print("点云视频生成失败")