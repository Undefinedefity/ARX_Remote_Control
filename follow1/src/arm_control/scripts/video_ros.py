#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import time
import numpy as np
from datetime import datetime

class RealsenseRecorder:
    def __init__(self, topic_name, output_dir="recordings", fps=30):
        """
        初始化RealSense录像器
        
        参数:
        topic_name: 订阅的RGB图像话题名称
        output_dir: 视频保存目录
        fps: 录制帧率
        """
        self.topic_name = topic_name
        self.output_dir = output_dir
        self.fps = fps
        self.bridge = CvBridge()
        self.video_writer = None
        self.is_recording = False
        self.start_time = None
        self.frame_count = 0
        self.width = None
        self.height = None
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化ROS节点
        rospy.init_node('realsense_recorder', anonymous=True)
        
        # 订阅指定话题
        self.image_sub = rospy.Subscriber(self.topic_name, Image, self.image_callback)
        
        print(f"订阅话题: {topic_name}")
        print("等待接收图像数据...")
    
    def image_callback(self, data):
        """图像数据回调函数"""
        try:
            # 将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # 记录第一个图像的分辨率
            if self.width is None:
                self.height, self.width = cv_image.shape[:2]
                print(f"检测到分辨率: {self.width}x{self.height}")
                
            # 显示预览
            cv2.imshow("RealSense Preview", cv_image)
            
            # 按's'键开始/停止录制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.toggle_recording()
            elif key == ord('q'):
                self.stop_recording()
                rospy.signal_shutdown("用户退出")
            
            # 如果正在录制，写入视频帧
            if self.is_recording:
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                
                # 在图像上添加录制信息
                cv2.putText(cv_image, f"REC {int(elapsed)}s | Frames: {self.frame_count}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 写入视频帧
                if self.video_writer:
                    self.video_writer.write(cv_image)
                
        except CvBridgeError as e:
            print(f"CV转换错误: {e}")
    
    def toggle_recording(self):
        """切换录制状态"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        """开始录制"""
        if self.width is None:
            print("尚未收到图像数据，无法开始录制")
            return
            
        # 创建唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join(self.output_dir, f"realsense_{timestamp}.mp4")
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'X264'/'avc1'
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        if not self.video_writer.isOpened():
            print(f"无法创建视频文件: {self.output_path}")
            return
            
        self.is_recording = True
        self.start_time = time.time()
        self.frame_count = 0
        print(f"开始录制: {self.output_path}")
    
    def stop_recording(self):
        """停止录制"""
        if self.is_recording and self.video_writer:
            self.is_recording = False
            self.video_writer.release()
            self.video_writer = None
            elapsed = time.time() - self.start_time
            print(f"录制完成: {self.output_path}")
            print(f"总帧数: {self.frame_count} | 时长: {elapsed:.2f}秒 | 平均FPS: {self.frame_count/elapsed:.1f}")
    
    def spin(self):
        """主循环"""
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("用户中断")
        finally:
            self.stop_recording()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # 默认话题名称 - 根据你的配置可能需要修改
    topic = "/mid_camera"  # RealSense默认彩色图像话题
    
    # 创建录像器并运行
    recorder = RealsenseRecorder(topic, output_dir="/home/arxpro/ARX_Remote_Control/data/video")
    recorder.spin()