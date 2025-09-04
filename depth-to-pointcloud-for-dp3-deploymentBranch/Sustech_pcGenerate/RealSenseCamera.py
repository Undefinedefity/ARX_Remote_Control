#AI generated code
import open3d as o3d
import numpy as np
import pyrealsense2 as rs

class RealSenseCamera:
    def __init__(self):
        # 初始化RealSense管道
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)
        
        # 获取深度参数
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        
        # 创建Open3D相机参数
        self.o3d_cam = o3d.camera.PinholeCameraIntrinsic()
        intr = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.o3d_cam.set_intrinsics(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
        
        # 坐标系转换矩阵（示例）
        self.transform = np.array([[1,0,0,0],
                                 [0,-1,0,0],
                                 [0,0,-1,0],
                                 [0,0,0,1]])

    def get_depth_frame(self):
        # 获取对齐后的深度帧
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        return np.asarray(depth_frame.get_data())

    def create_pointcloud(self, depth_frame):
        # 转换为Open3D格式
        depth_image = o3d.geometry.Image(depth_frame)
        
        # 创建点云
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image, 
            self.o3d_cam,
            depth_scale=1/self.depth_scale,
            project_valid_depth_only=True
        )
        
        # 应用坐标系变换
        pcd.transform(self.transform)
        return pcd

# 使用示例
if __name__ == "__main__":
    cam = RealSenseCamera()
    while True:
        depth = cam.get_depth_frame()
        pcd = cam.create_pointcloud(depth)
        
        # 可视化或处理点云
        o3d.visualization.draw_geometries([pcd])