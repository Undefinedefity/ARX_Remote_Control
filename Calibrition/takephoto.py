import os
import cv2
import numpy as np
import pyrealsense2 as rs

# 配置参数
SAVE_DIR = "Calibrition/calib_images1"  # 保存目录
PATTERN_SIZE = (7, 7)           # 棋盘格内角点数
SQUARE_SIZE = 15                # 棋盘格实际尺寸（毫米）
MIN_IMAGES = 20                 # 最少需要拍摄的图像数量
ANALYZE_IMMEDIATELY = False    # 是否立即分析拍摄的图像（False表示最后统一分析）

def init_camera():
    """初始化RealSense相机"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
        print("摄像头已成功初始化。")
        return pipeline
    except Exception as e:
        print(f"无法初始化摄像头: {e}")
        raise

def capture_chessboard_images(pipeline, save_dir):
    """拍摄标定板图像，并在界面显示是否检测到标定板"""
    os.makedirs(save_dir, exist_ok=True)
    print(f"图像将保存到: {os.path.abspath(save_dir)}")
    
    img_count = 0
    cv2.namedWindow("Calibration Capture", cv2.WINDOW_AUTOSIZE)
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            color_image = np.asanyarray(color_frame.get_data())
            
            # 检测棋盘格角点
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)
            
            # 在图像上显示检测结果
            display_img = color_image.copy()
            if ret:
                # 检测到标定板
                cv2.drawChessboardCorners(display_img, PATTERN_SIZE, corners, ret)
                status_text = "标定板已检测到! 按 's' 保存"
                color_status = (0, 255, 0)  # 绿色
            else:
                status_text = "未检测到标定板，请调整角度"
                color_status = (0, 0, 255)  # 红色
            
            # 在图像顶部显示状态信息
            cv2.rectangle(display_img, (0, 0), (640, 40), color_status, -1)
            cv2.putText(display_img, status_text, (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow("Calibration Capture", display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC退出
                print("用户按下ESC键，退出程序。")
                break
            elif key == ord('s') and ret:  # 只有检测到标定板时才能保存
                filename = os.path.join(save_dir, f"chessboard_{img_count:03d}.png")
                success = cv2.imwrite(filename, color_image)
                if success:
                    print(f"Saved: {filename}")
                    img_count += 1
                    # 显示保存成功提示
                    vis = color_image.copy()
                    vis[:60, :] = (0, 255, 0)  # 顶部绿色背景
                    cv2.putText(vis, "CAPTURE SUCCESS!", (150, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    cv2.imshow("Calibration Capture", vis)
                    cv2.waitKey(500)
                else:
                    print(f"Failed to save image: {filename}")
    
    finally:
        if not ANALYZE_IMMEDIATELY:
            pipeline.stop()
            cv2.destroyAllWindows()
            print(f"\n拍摄完成！共获取 {img_count} 张有效图像")
        return img_count

def main():
    # 初始化相机
    try:
        pipe = init_camera()
        
        # 拍摄图像
        print("\n拍摄操作指南：")
        print("1. 确保棋盘格充满画面至少50%")
        print("2. 不同角度倾斜棋盘格（±45°）")
        print("3. 当画面显示时，按 's' 键拍摄（仅在检测到标定板时可用）")
        print(f"4. 获得足够图像后按ESC退出")
        
        img_count = capture_chessboard_images(pipe, SAVE_DIR)
        
        if img_count < MIN_IMAGES:
            print(f"\n警告：只获取了 {img_count} 张图像，建议至少 {MIN_IMAGES} 张以获得更好的标定结果。")
        
    except Exception as e:
        print(f"程序发生错误: {e}")
    finally:
        if 'pipe' in locals() and 'pipe' in globals():
            pipe.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
