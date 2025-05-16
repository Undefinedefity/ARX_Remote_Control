#!/home/dc/anaconda3/envs/dc/bin/python
import time
import rospy
import sys
import threading
from message_filters import ApproximateTimeSynchronizer, Subscriber
sys.path.append("/home/dc/anaconda3/envs/dc/lib/python3.8/site-packages")
import numpy as np
import cv2
import zarr
import os
from cv_bridge import CvBridge
from arm_control.msg import JointInformation, JointControl, PosCmd
from sensor_msgs.msg import Image

# 全局配置参数
TOTAL_EPISODES = 40         # 需要采集的episode总数
STEPS_PER_EPISODE = 150    # 每个episode的步长
DATA_ROOT = '/home/arxpro/ARX_Remote_Control/data/multi_episodes'  # 数据存储根目录

# 全局状态变量
data_lock = threading.Lock()
current_episode = 0
current_step = 0
collecting_data = True
all_episodes = []

# 数据存储结构
data_template = {
    'qpos': [],
    'action': [],
    'eef_qpos': [],
    'images_mid': [],
    'images_right': [],
    'depth': []
}

def init_data_directory():
    """初始化数据存储目录结构"""
    os.makedirs(os.path.join(DATA_ROOT, 'videos'), exist_ok=True)
    return zarr.open(os.path.join(DATA_ROOT, 'dataset.zarr'), mode='w')

def save_episode_video(episode_data, episode_idx):
    """保存单个episode的视频记录"""
    video_path = os.path.join(DATA_ROOT, 'videos', f'episode_{episode_idx}.mp4')
    merged_frames = np.concatenate([episode_data['images_mid'], episode_data['images_right']], axis=2)
    
    video = cv2.VideoWriter(
        video_path, 
        cv2.VideoWriter_fourcc(*'mp4v'),
        10, 
        (merged_frames.shape[2], merged_frames.shape[1])
    )
    
    for frame in merged_frames:
        video.write(frame)
    video.release()

def finalize_dataset(zarr_root, all_data):
    """将全部episode数据写入Zarr存储"""
    # 创建数据存储结构
    observations = zarr_root.create_group('observations')
    images = observations.create_group('images')
    
    # 合并所有episode数据
    merged = {
        'qpos': np.concatenate([ep['qpos'] for ep in all_data]),
        'action': np.concatenate([ep['action'] for ep in all_data]),
        'eef_qpos': np.concatenate([ep['eef_qpos'] for ep in all_data]),
        'images_mid': np.concatenate([ep['images_mid'] for ep in all_data]),
        'images_right': np.concatenate([ep['images_right'] for ep in all_data]),
        'depth': np.concatenate([ep['depth'] for ep in all_data])
    }pip3 install open3d -i https://pypi.tuna.tsinghua.edu.cn/simple

    # 配置Zarr存储参数
    store_config = {
        'images_mid': {
            'shape': merged['images_mid'].shape,
            'dtype': 'uint8',
            'chunks': (50, 480, 640, 3)
        },
        'images_right': {
            'shape': merged['images_right'].shape,
            'dtype': 'uint8',
            'chunks': (50, 480, 640, 3)
        },
        'depth': {
            'shape': merged['depth'].shape,
            'dtype': 'uint16',
            'chunks': (50, 480, 640)
        }
    }

    # 写入数据
    for key in ['qpos', 'action', 'eef_qpos']:
        zarr_root.array(key, merged[key], dtype='float64')
    
    images.array('mid', merged['images_mid'], **store_config['images_mid'])
    images.array('right', merged['images_right'], **store_config['images_right'])
    observations.array('depth', merged['depth'], **store_config['depth'])

def data_collection_callback(joint_ctrl, joint_info, eef_pos, img_mid, img_right, depth_img):
    """多传感器数据采集回调函数"""
    global current_step, collecting_data, all_episodes, current_episode
    
    with data_lock:
        if not collecting_data:
            return

        bridge = CvBridge()
        
        # 转换图像数据
        data_template['images_mid'].append(bridge.imgmsg_to_cv2(img_mid, "bgr8"))
        data_template['images_right'].append(bridge.imgmsg_to_cv2(img_right, "bgr8"))
        data_template['depth'].append(bridge.imgmsg_to_cv2(depth_img, "16UC1"))
        
        # 记录运动数据
        data_template['qpos'].append(joint_info.joint_pos)
        data_template['action'].append(joint_ctrl.joint_pos)
        data_template['eef_qpos'].append([
            eef_pos.x, eef_pos.y, eef_pos.z,
            eef_pos.roll, eef_pos.pitch, eef_pos.yaw,
            eef_pos.gripper
        ])
        
        current_step += 1
        print("Step:",current_step)

        # Episode完成检测
        if current_step >= STEPS_PER_EPISODE:
            collecting_data = False
            print(f'Episode {current_episode} end__________________________________')
            
            # 转换数据格式并保存
            episode_data = {
                k: np.array(v) for k, v in data_template.items()
            }
            
            # 重置采集状态
            data_template.update({k: [] for k in data_template})
            current_step = 0
            
            # 异步保存视频
            save_episode_video(episode_data, current_episode)
            all_episodes.append(episode_data)

            print("Sleep 10.0 s")
            rospy.sleep(10.0)
            print("Wake Up")
            
            # 更新episode计数
            current_episode += 1
            print(f"Current Episode:{current_episode}")
            
            if current_episode < TOTAL_EPISODES:
                collecting_data = True
            else:
                # 最终数据存储
                zarr_root = init_data_directory()
                finalize_dataset(zarr_root, all_episodes)
                rospy.signal_shutdown("数据采集完成")

if __name__ == "__main__":
    # ROS节点初始化
    rospy.init_node("multi_episode_collector")
    
    # 初始化Zarr存储
    init_data_directory()

    # 创建数据订阅器
    sync = ApproximateTimeSynchronizer([
        Subscriber("joint_control", JointControl),
        Subscriber("joint_information", JointInformation),
        Subscriber("follow1_pos_back", PosCmd),
        Subscriber("mid_camera", Image),
        Subscriber("right_camera", Image),
        Subscriber("mid_depth_camera", Image)
    ], slop=0.1, queue_size=20)
    
    sync.registerCallback(data_collection_callback)
    
    # 主循环
    while not rospy.is_shutdown():
        rospy.spin()
        
    print("所有episode数据采集存储完成！")