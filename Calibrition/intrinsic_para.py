import pyrealsense2 as rs
import numpy as np

# 初始化相机管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 彩色流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)    # 深度流

# 启动设备
profile = pipeline.start(config)

# 获取彩色相机的内参
color_profile = profile.get_stream(rs.stream.color)
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

# 打印内参
print("彩色相机内参：")
print(f"焦距 (fx, fy): ({color_intrinsics.fx}, {color_intrinsics.fy})")
print(f"主点 (cx, cy): ({color_intrinsics.ppx}, {color_intrinsics.ppy})")
print(f"畸变模型: {color_intrinsics.model}")
print(f"畸变系数: {np.array(color_intrinsics.coeffs)}")

# 深度相机的内参（通常与彩色相机不同）
depth_profile = profile.get_stream(rs.stream.depth)
depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
print("\n深度相机内参：")
print(f"焦距 (fx, fy): ({depth_intrinsics.fx}, {depth_intrinsics.fy})")
print(f"主点 (cx, cy): ({depth_intrinsics.ppx}, {depth_intrinsics.ppy})")

# 停止管道
pipeline.stop()