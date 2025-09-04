import zarr
import numpy as np
import cv2
import os
from tqdm import tqdm

def zarr_to_video(zarr_path, output_video, fps=30, duration_sec=10):
    """
    将Zarr文件中的RGB图像序列转换为视频
    
    参数:
    zarr_path: Zarr存储路径
    output_video: 输出视频文件路径
    fps: 输出视频的帧率
    duration_sec: 期望的视频时长(秒)
    """
    # 读取Zarr数组
    store = zarr.DirectoryStore(zarr_path)
    zarr_array = zarr.open(store, mode='r')
    
    # 确保Zarr数组是3D或4D
    if zarr_array.ndim not in [3, 4]:
        raise ValueError("Zarr数组必须是3D (H, W, C) 或4D (T, H, W, C)")
    
    # 获取视频参数
    is_sequence = zarr_array.ndim == 4
    total_frames = zarr_array.shape[0] if is_sequence else 1
    
    # 计算所需的帧数（根据期望时长和帧率）
    desired_frames = int(fps * duration_sec)
    
    # 设置实际要处理的帧数（不超过总帧数）
    frame_count = min(desired_frames, total_frames) if is_sequence else 1
    actual_duration = frame_count / fps
    
    height, width = zarr_array.shape[1:3] if is_sequence else zarr_array.shape[0:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        raise RuntimeError("无法创建视频文件，请检查编解码器或输出路径")
    
    # 添加帧数限制提示
    frame_info = f"总帧数: {total_frames} | 截取帧数: {frame_count}"
    duration_info = f"时长: {actual_duration:.2f}秒"
    print(f"正在处理: {frame_info} | {duration_info} | 分辨率: {width}x{height} | 帧率: {fps}FPS")
    
    # 逐帧处理并写入视频
    with tqdm(total=frame_count, desc="生成视频") as pbar:
        for i in range(frame_count):
            # 获取当前帧
            frame = zarr_array[i] if is_sequence else zarr_array[:]
            
            # 确保数据类型为uint8
            if frame.dtype != np.uint8:
                if np.max(frame) <= 1:  # 浮点型[0,1]转[0,255]
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # 处理单通道图像
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # 处理RGB图像 (Zarr通常存储为Channel Last)
            elif frame.ndim == 3:
                if frame.shape[-1] == 4:  # 处理RGBA图像
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[-1] == 3:  # RGB转BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif frame.shape[-1] == 1:  # 单通道扩展
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                raise ValueError("不支持的通道格式")
                
            video_writer.write(frame)
            pbar.update(1)
    
    # 释放资源
    video_writer.release()
    print(f"视频已保存至: {os.path.abspath(output_video)}")
    print(f"实际时长: {actual_duration:.2f}秒 | 帧率: {fps}FPS | 总帧数: {frame_count}")

if __name__ == "__main__":
    # 示例用法
    zarr_path = "/home/arxpro/ARX_Remote_Control/data/9_3_lemon_plate_1.zarr/data/img_mid"
    output_video = "/home/arxpro/ARX_Remote_Control/data/9_3_lemon_plate_1.zarr/data/rbg_vid.mp4"

    # 设置10秒时长
    zarr_to_video(zarr_path, output_video, fps=10, duration_sec=15)