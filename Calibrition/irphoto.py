import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

def capture_ir_images(save_folder, image_count=20, resolution=(640, 480), fps=30):
    """
    修正版D435i IR图像采集
    :param save_folder: 图像保存目录
    :param image_count: 需要捕获的图像数量
    :param resolution: 图像分辨率（宽, 高）
    :param fps: 采集帧率
    """
    # 创建保存目录
    os.makedirs(save_folder, exist_ok=True)
    
    # 配置管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 修正参数顺序：stream_type, stream_index, width, height, format, framerate
    config.enable_stream(rs.stream.infrared, 1,  # 左IR摄像头
                        resolution[0], resolution[1],
                        rs.format.y8, fps)

    try:
        # 启动流
        pipeline.start(config)
        print("红外摄像头已启动...")
        print("操作指南：")
        print("- 按 [空格] 保存当前图像")
        print("- 按 [q] 退出采集")

        saved_count = 0
        while saved_count < image_count:
            # 等待帧数据（增加超时时间）
            frames = pipeline.wait_for_frames(5000)  # 5秒超时
            
            # 获取左IR帧
            ir_frame = frames.get_infrared_frame(1)
            ir_image = np.asanyarray(ir_frame.get_data())
            
            # 显示图像（优化显示比例）
            display_image = cv2.resize(ir_image, (resolution[0]//2, resolution[1]//2))
            cv2.imshow('IR Preview', display_image)
            
            # 键盘控制（优化响应速度）
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_folder, f"ir_{timestamp}_{saved_count:03d}.png")
                cv2.imwrite(filename, ir_image)
                print(f"已保存：{filename}")
                saved_count += 1
            elif key == ord('q'):
                break

    except Exception as e:
        print(f"运行时错误：{str(e)}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 依赖检查
    try:
        rs.context()
    except:
        print("请先安装pyrealsense2库：")
        print("pip install pyrealsense2")
        exit()

    # 运行采集
    capture_ir_images(
        save_folder="calibrition/ir_images1",
        image_count=30,
        resolution=(848, 480),  # 可选分辨率：(848, 480) 或 (640, 480)
        fps=15  # 高分辨率建议使用较低帧率
    )
