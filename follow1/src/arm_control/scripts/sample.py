#!/home/dc/anaconda3/envs/dc/bin/python
import time
import rospy
import sys
from message_filters import ApproximateTimeSynchronizer,Subscriber
sys.path.append("/home/dc/anaconda3/envs/dc/lib/python3.8/site-packages")
import numpy as np
import cv2
import h5py
import zarr
from cv_bridge import CvBridge
from arm_control.msg import JointInformation
from arm_control.msg import JointControl
from arm_control.msg import PosCmd
from sensor_msgs.msg import Image
import os

def count_files_with_extension(directory, extension):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count

global data_dict, step, Max_step, dataset_path 

# parameters
step = 0
Max_step = 200 #1000
# directory_path = f'/media/dc/CLEAR/xgxy/dataset20241213' # f'/media/dc/ESD-USB/1120-remote-data'# f'/media/dc/HP2024/data/SCIL/Task4_long_horizon'

directory_path = f'/home/arxpro/ARX_Remote_Control/data/5_12' # f'/media/dc/ESD-USB/1120-remote-data'# f'/media/dc/HP2024/data/SCIL/Task4_long_horizon'
extension = '.zarr' 
episode_idx = count_files_with_extension(directory_path, extension)
dataset_path = f'{directory_path}/episode_{episode_idx}.zarr'
video_path=f'{directory_path}/video/{episode_idx}'
data_dict = {
        '/observations/qpos': [],
        '/action': [],
        '/eef_qpos': [],
        '/observations/images/mid' : [],
        '/observations/images/right' : [],
        '/observations/depth' : [],
        }


def callback(JointCTR2,JointInfo2,f2p,image_mid,image_right,depth):
    global data_dict, step, Max_step, dataset_path,video_path
    print(f"DEBUG:Enter Callback!")
    save=True
    bridge = CvBridge()
    image_mid = bridge.imgmsg_to_cv2(image_mid, "bgr8")
    image_right = bridge.imgmsg_to_cv2(image_right, "bgr8")
    depth = bridge.imgmsg_to_cv2(depth, "16UC1")
    eef_qpos=np.array([f2p.x,f2p.y,f2p.z,f2p.roll,f2p.pitch,f2p.yaw,f2p.gripper])
    action = np.array(JointCTR2.joint_pos)
    qpos =np.array(JointInfo2.joint_pos)
    # print("eef_qpos:", eef_qpos)
    # print("action:", action)
    if save:
        data_dict["/eef_qpos"].append(eef_qpos)
        data_dict["/action"].append(action)
        data_dict["/observations/qpos"].append(qpos)
        data_dict["/observations/images/mid"].append(image_mid)
        data_dict["/observations/images/right"].append(image_right)
        data_dict["/observations/depth"].append(depth)
        print(f"[DEBUG] Saved")

    canvas = np.zeros((480, 1280, 3), dtype=np.uint8)

    # 将图像复制到画布的特定位置
    # canvas[:, :640, :] = image_left
    # canvas[:, 640:1280, :] = image_mid
    # canvas[:, 1280:, :] = image_right
    canvas[:, :640, :] = image_mid
    canvas[:, 640:1280, :] = image_right

    # 在一个窗口中显示排列后的图像
    cv2.imshow('Multi Camera Viewer', canvas)
  
    cv2.waitKey(1)

    step = step + 1
    print(step)
    if step >= Max_step and save:
        print('end__________________________________')
        
        # 创建 Zarr 存储（自动创建目录结构）
        store = zarr.DirectoryStore(dataset_path)  # Zarr 使用目录存储
        root = zarr.group(store=store, overwrite=True)
        
        # 设置属性
        root.attrs['sim'] = True
        
        # 创建 observations 组
        obs = root.create_group('observations')
        image = obs.create_group('images')
        
        # 创建数据集
        image.create_dataset('mid', 
                            data=data_dict['/observations/images/mid'],
                            shape=(Max_step, 480, 640, 3),
                            dtype='uint8',
                            chunks=(1, 480, 640, 3))  # 保持相同分块
        
        image.create_dataset('right',
                            data=data_dict['/observations/images/right'],
                            shape=(Max_step, 480, 640, 3),
                            dtype='uint8',
                            chunks=(1, 480, 640, 3))
        
        obs.create_dataset('depth',
                            data=data_dict['/observations/depth'],
                            shape=(Max_step, 480, 640),
                            dtype='uint16',
                            chunks=(1, 480, 640))
        
        obs.create_dataset('qpos', data=data_dict['/observations/qpos'])

        root.create_dataset('action', data=data_dict['/action'])

        root.create_dataset('eef_qpos', data=data_dict['/eef_qpos'])
        
        # 视频生成部分保持不变（直接从 data_dict 读取）
        mid_images = data_dict['/observations/images/mid']
        right_images = data_dict['/observations/images/right']
        images = np.concatenate([mid_images, right_images], axis=2)
        
        video_path = f'{video_path}video.mp4'
        height, width, _ = images[0].shape
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        for img in images:
            video_writer.write(img)
        video_writer.release()
        
        print('end__________________________________')
        rospy.signal_shutdown("\n************************signal_shutdown********sample successfully!*************************************")
        quit("sample successfully!")
        

if __name__ =="__main__":
    #config my camera
    time.sleep(2)  # wait 2s to start
    
    rospy.init_node("My_node1")
    
    a=time.time()
    # master1_pos = Subscriber("master1_pos_back",PosCmd)
    # master2_pos = Subscriber("master2_pos_back",PosCmd)
    follow1_pos = Subscriber("follow1_pos_back",PosCmd)
    # follow2_pos = Subscriber("follow2_pos_back",PosCmd)
    master1 = Subscriber("joint_control",JointControl)
    # master2 = Subscriber("joint_control2",JointControl)
    follow1 = Subscriber("joint_information",JointInformation)
    # follow2 = Subscriber("joint_information2",JointInformation)
    image_mid = Subscriber("mid_camera",Image)
    # image_left = Subscriber("left_camera",Image)
    image_right = Subscriber("right_camera",Image)
    depth = Subscriber("mid_depth_camera",Image)
    ats = ApproximateTimeSynchronizer([master1,follow1,follow1_pos,image_mid,image_right,depth],slop=0.15,queue_size=40)
    print(f"Hello")
    ats.registerCallback(callback)
    rospy.spin()
    
