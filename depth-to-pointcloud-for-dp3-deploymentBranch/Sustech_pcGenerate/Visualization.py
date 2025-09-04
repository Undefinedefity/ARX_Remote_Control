import open3d as o3d
import numpy as np
import zarr
from Cloud_Process import boundary

# 加载Zarr数据
zarr_path = "/home/arxpro/ARX_Remote_Control/data/9_2_lemon_plate_2.zarr/data/pcd"
zarr_root = zarr.open(zarr_path, mode='r')
point_clouds = zarr_root['point_cloud']  # 形状 [N, 1024, 6]

# 选择帧（例如第1200帧）
frame_idx = 90
pc_data = point_clouds[frame_idx]

# 仅提取坐标（忽略颜色信息）
points = pc_data[:, :3]  # 只取xyz坐标

# 创建Open3D点云（无颜色）
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 坐标系（红色-X，绿色-Y，蓝色-Z）
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
#bounding box
WORK_SPACE = [
    [-0.05, 0.1],
    [-0.04, 0.2],
    [-0.0025, 0.1]
]
min_bound, max_bound = boundary(WORK_SPACE)
custom_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
custom_bbox.color = (0, 1, 0)  # 设置为绿色
# 创建可视化器
vis = o3d.visualization.Visualizer()
vis.create_window()

# 添加几何体
vis.add_geometry(pcd)
vis.add_geometry(custom_bbox)
vis.add_geometry(coord_frame)

# 设置灰色背景（RGB值0-1）
render_opt = vis.get_render_option()
render_opt.background_color = np.array([0.5, 0.5, 0.5])  # 中灰色
render_opt.point_size = 3.0  # 可选：调整点大小

# 运行可视化器
vis.run()
vis.destroy_window()