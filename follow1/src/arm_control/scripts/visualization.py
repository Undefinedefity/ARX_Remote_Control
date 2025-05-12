#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class DualImageViewer:
    def __init__(self):
        self.bridge = CvBridge()
        
        # 创建两个独立的窗口
        cv2.namedWindow("RGB Image Mid", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("RGB Image Right", cv2.WINDOW_NORMAL)

        # 订阅 RGB 和深度图话题（假设深度话题为 `/depth_camera`）
        rospy.Subscriber('/mid_camera', Image, self.rgb_callback_mid)
        rospy.Subscriber('/mid_depth_camera', Image, self.depth_callback)
        rospy.Subscriber('/right_camera', Image, self.rgb_callback_right)

    def rgb_callback_mid(self, msg):
        try:
            # 转换 RGB 图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("RGB Image Mid", cv_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr(e)

    def rgb_callback_right(self, msg):
        try:
            # 转换 RGB 图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("RGB Image Right", cv_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr(e)

    def depth_callback(self, msg):
        try:
            # 转换深度图（假设深度图为 16UC1 或 32FC1 格式）
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            # 将深度图归一化到 0-255 范围（可视化更清晰）
            if depth_image.dtype == np.uint16:
                depth_image = cv2.convertScaleAbs(depth_image, alpha=0.05)  # 16UC1 缩放
            elif depth_image.dtype == np.float32:
                depth_image = np.nan_to_num(depth_image)  # 处理 NaN 值
                depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # 应用颜色映射（可选）
            depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            cv2.imshow("Depth Image", depth_colormap)
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    rospy.init_node('dual_image_viewer')
    div = DualImageViewer()
    rospy.spin()
    cv2.destroyAllWindows()