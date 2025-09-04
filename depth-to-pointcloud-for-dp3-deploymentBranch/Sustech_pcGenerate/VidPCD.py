import open3d as o3d
import numpy as np
import zarr
import cv2
import os
from tqdm import tqdm
from Cloud_Process import boundary

# 配置参数
ZARR_PATH = "/home/arxpro/ARX_Remote_Control/data/9_2_lemon_plate_2.zarr/data/pcd"
OUTPUT_VIDEO = "/home/arxpro/ARX_Remote_Control/data/9_2_lemon_plate_2.zarr/data/point_cloud_video.mp4"
FRAME_RATE = 30  # 视频帧率
START_FRAME = 0
END_FRAME = 4000  # 结束帧索引
TEMP_DIR = "/home/arxpro/temp_frames"  # 临时保存图像的目录

# 创建工作空间边界框
WORK_SPACE = [
    [-0.05, 0.1],
    [-0.04, 0.2],
    [-0.0025, 0.1]
]
min_bound, max_bound = boundary(WORK_SPACE)
custom_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
custom_bbox.color = (0, 1, 0)  # 绿色

# 创建坐标系
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

# 加载Zarr数据
zarr_root = zarr.open(ZARR_PATH, mode='r')
point_clouds = zarr_root['pointcloud']  # 形状 [N, 1024, 6]
total_frames = min(END_FRAME - START_FRAME + 1, len(point_clouds))

# 创建临时目录
os.makedirs(TEMP_DIR, exist_ok=True)

# 初始化可视化器
vis = o3d.visualization.Visualizer()
vis.create_window(width=1280, height=720, visible=True)  # 不可见模式提高性能
vis.add_geometry(custom_bbox)
vis.add_geometry(coord_frame)

# 添加初始点云（空点云）
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

# 设置渲染选项
render_opt = vis.get_render_option()
render_opt.background_color = np.array([0.5, 0.5, 0.5])  # 灰色背景
render_opt.point_size = 3.0

# 获取并固定相机视角（使用第一帧的视角）
first_frame = point_clouds[START_FRAME][:, :3]
pcd.points = o3d.utility.Vector3dVector(first_frame)
vis.update_geometry(pcd)
vis.poll_events()
vis.update_renderer()
view_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

# 逐帧渲染并保存图像
print(f"Rendering {total_frames} frames...")
for i in tqdm(range(START_FRAME, min(END_FRAME + 1, len(point_clouds)))):
    # 更新点云数据
    points = point_clouds[i][:, :3]
    pcd.points = o3d.utility.Vector3dVector(points)
    vis.update_geometry(pcd)
    
    # 保持相机视角一致
    vis.get_view_control().convert_from_pinhole_camera_parameters(view_params)
    
    # 渲染并保存图像
    vis.poll_events()
    vis.update_renderer()
    image_path = os.path.join(TEMP_DIR, f"frame_{i:06d}.png")
    vis.capture_screen_image(image_path)

# 清理Open3D资源
vis.destroy_window()

# 将图像序列编码为视频
print("Creating video from frames...")
images = [img for img in os.listdir(TEMP_DIR) if img.endswith(".png")]
images.sort()

# 获取第一帧的尺寸
frame = cv2.imread(os.path.join(TEMP_DIR, images[0]))
height, width, _ = frame.shape

# 初始化视频编码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码
video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FRAME_RATE, (width, height))

# 逐帧写入视频
for image_name in tqdm(images):
    image_path = os.path.join(TEMP_DIR, image_name)
    frame = cv2.imread(image_path)
    video.write(frame)

# 释放资源
video.release()

# 清理临时文件
print("Cleaning up temporary files...")
for image_name in images:
    os.remove(os.path.join(TEMP_DIR, image_name))
os.rmdir(TEMP_DIR)

print(f"Video saved to: {OUTPUT_VIDEO}")