import os
import cv2
import numpy as np
import pyrealsense2 as rs
#这里import另一个代码框架的内容
import sys
sys.path.append('/home/arxpro/ARX_Remote_Control/depth-to-pointcloud-for-dp3-deploymentBranch/Sustech_pcGenerate')
from Cloud_Process import farthest_point_sampling,preprocess_point_cloud
from Convert_PointCloud import PointCloudGenerator

# 使用示例
generator = PointCloudGenerator()

# 配置参数
SAVE_DIR = "depth_images"        # 深度图像保存目录
DEPTH_FORMAT = rs.format.z16      # 深度数据格式
DEPTH_RES = (640, 480)           # 深度分辨率
FPS = 30                         # 帧率
VIS_SCALE = 0.5                  # 深度可视化缩放系数 (调整可视效果)

def init_depth_camera():
    """初始化深度相机"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 配置深度流
    config.enable_stream(rs.stream.depth, DEPTH_RES[0], DEPTH_RES[1], DEPTH_FORMAT, FPS)
    
    try:
        pipeline.start(config)
        print("深度相机已成功初始化。")
        return pipeline
    except Exception as e:
        print(f"无法初始化深度相机: {e}")
        raise

def visualize_depth(depth_frame):
    """将深度帧转换为可视化的伪彩色图像"""
    depth_image = np.asanyarray(depth_frame.get_data())
    
    # 转换为8位灰度图像
    depth_colormap = cv2.convertScaleAbs(depth_image, alpha=VIS_SCALE)
    
    # 应用伪彩色映射（JET风格）
    depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)
    return depth_colormap

def capture_depth_images(pipeline, save_dir):
    """实时捕获并显示深度图像，按s键保存"""
    os.makedirs(save_dir, exist_ok=True)
    print(f"深度图像将保存到: {os.path.abspath(save_dir)}")
    
    img_count = 0
    cv2.namedWindow("Depth Capture", cv2.WINDOW_AUTOSIZE)
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            
            # 转换为可视图像
            vis_image = visualize_depth(depth_frame)
            # 获取点云数组
            depth_image = np.asanyarray(depth_frame.get_data())
            return depth_image
        #ZYT:其实现在不需要可视化的功能，可以留着作为验证
            
            # # 显示状态信息
            # display_img = vis_image.copy()
            # status_text = f"已保存: {img_count} 按 's' 保存 | ESC退出"
            # cv2.putText(display_img, status_text, (10, 30), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # cv2.imshow("Depth Capture", display_img)
            
            # key = cv2.waitKey(1) & 0xFF
            # if key == 27:  # ESC退出
            #     print("用户按下ESC键，退出程序。")
            #     break
            # elif key == ord('s'):
            #     # 保存原始深度数据（16位PNG）
            #     depth_image = np.asanyarray(depth_frame.get_data())
            #     filename = os.path.join(save_dir, f"depth_{img_count:03d}.png")
            #     cv2.imwrite(filename, depth_image)
                
            #     # 保存可视化版本（可选）
            #     vis_filename = os.path.join(save_dir, f"depth_vis_{img_count:03d}.png")
            #     cv2.imwrite(vis_filename, vis_image)
                
            #     print(f"Saved: {filename} 和可视化版本 {vis_filename}")
            #     img_count += 1
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n采集完成！共保存 {img_count} 张深度图像")

def depth2Pcd(pipeline,SAVE_DIR):
    depth_from_robot = capture_depth_images(pipeline, SAVE_DIR)
    print(f"深度数据形状: {depth_from_robot.shape}, 数据类型: {depth_from_robot.dtype}")

    pc_generator = PointCloudGenerator(
        img_size=depth_from_robot.shape[1]  # 使用深度图高度
    )
    print("点云生成器初始化完成")
    
    # 检查深度图有效性
    print(f"深度图范围: {np.min(depth_from_robot)} - {np.max(depth_from_robot)}")
    if np.all(depth_from_robot == 0):
        print("警告: 深度图全为零值")
        return None

    # 生成点云（添加详细日志）
    print("正在生成点云...")
    points, _ = pc_generator.generateCroppedPointCloud(
        depth_data=depth_from_robot,
    )
    print(f"生成点云形状: {points.shape if isinstance(points, np.ndarray) else '无效'}")
    
    if not isinstance(points, np.ndarray) or points.size == 0:
        raise ValueError("生成的点云为空")

    sampled_points = preprocess_point_cloud(points)  # 确保输入为NumPy数组
    print(f"处理后点云形状: {sampled_points.shape}")

    # 验证最终输出
    if sampled_points.shape == (1024, 3):
        print("correct shape: (1024, 3)")
    else:
        print(f"警告: 无效的输出形状 {sampled_points.shape}")
    return sampled_points





def main():
    try:
        pipe = init_depth_camera()
        
        print("\n操作指南：")
        print("1. 确保深度相机对准目标场景")
        print("2. 实时画面会显示伪彩色深度图")
        print("3. 按 's' 键保存当前深度帧（原始数据+可视化版本）")
        print("4. 按ESC键退出程序")
        
        capture_depth_images(pipe, SAVE_DIR)
        pointcloud = depth2Pcd(pipe,SAVE_DIR)
        
    except Exception as e:
        print(f"程序发生错误: {e}")

if __name__ == "__main__":
    main()
