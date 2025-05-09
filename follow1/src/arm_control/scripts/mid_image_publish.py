import sys
sys.path.append("/home/dc/anaconda3/envs/dc/lib/python3.8/site-packages")
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np

# 配置相机
# ['317622074564', '233622076982', '234322070358']
def configure_camera(serial_number):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  
    pipeline.start(config)
    return pipeline

def main():
    # print out all available camera
    # ctx = rs.context()
    # for d in ctx.devices:
    #     print("found device: ", d)
    # print("after for loop")

    # 获取连接的 RealSense 相机序列号
    serial_number_dict = {'mid_white':'827312072685', 'left_black':'317622074564', 'right_white': '317622070255'}
    # serial_number_list = ['207222070486', '233622076982', '234322070358', '827312072685']         #827312072685: Top 317622074564:Middle Camera  233622076982:Right Camera  234322070358:Left Camera
    Mid_pipeline = configure_camera(serial_number_dict['mid_white'])
    # Mid_pipeline = configure_camera(serial_number_dict['side_view'])

    # Mid_pipeline = configure_camera('207222070486')

    # 初始化ROS节点
    rospy.init_node('mid_camera_publisher')

    # 创建CvBridge对象，用于ROS图像消息和OpenCV图像格式之间的转换
    bridge = CvBridge()

    # 创建一个发布者，发布图像到指定的ROS话题
    image_pub = rospy.Publisher('/mid_camera', Image, queue_size=10)


    # 循环发布图像
    rate = rospy.Rate(10)  # 设置发布频率为10Hz
    while not rospy.is_shutdown():
        frames = Mid_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame is None:
            continue
        color_data = np.asanyarray(color_frame.get_data())
        # 将OpenCV格式的图像转换为ROS图像消息
        image_msg = bridge.cv2_to_imgmsg(color_data, encoding="bgr8")
        # 发布图像消息到指定的ROS话题
        image_msg.header.stamp = rospy.Time.now()
        image_pub.publish(image_msg)

        rate.sleep()

    # 停止摄像头并关闭窗口
    Mid_pipeline.stop() 

if __name__ == '__main__':

    main()


