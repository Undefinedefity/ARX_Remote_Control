# import sys
# sys.path.append("/home/dc/anaconda3/envs/dc/lib/python3.8/site-packages")
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np

# 配置相机深度流
def configure_camera(serial_number):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    # 启用深度流（640x480，Z16格式，30fps）
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    return pipeline

def main():
    serial_number_dict = {
        'mid_white': '827312072685',
        'left_black': '317622074564', 
        'right_white': '317622070255'
    }
    
    # 初始化深度相机
    Mid_pipeline = configure_camera(serial_number_dict['mid_white'])

    # 初始化ROS节点
    rospy.init_node('mid_depth_camera_publisher')
    bridge = CvBridge()
    
    # 创建深度图像发布者
    depth_pub = rospy.Publisher('/mid_depth_camera', Image, queue_size=10)

    rate = rospy.Rate(10)  # 发布频率10Hz
    while not rospy.is_shutdown():
        frames = Mid_pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            continue
            
        # 将深度数据转换为numpy数组
        depth_data = np.asanyarray(depth_frame.get_data())
        
        # 创建ROS图像消息（16UC1编码对应深度数据）
        depth_msg = bridge.cv2_to_imgmsg(depth_data, encoding="16UC1")
        depth_msg.header.stamp = rospy.Time.now()
        depth_pub.publish(depth_msg)
        
        rate.sleep()

    Mid_pipeline.stop()

if __name__ == '__main__':
    main()