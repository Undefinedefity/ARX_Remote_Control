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
import copy
from termcolor import cprint

global data_dict, sub_step, step, Max_step, Max_episode, dataset_path,episode_ends_array,mid_image_array,right_image_array,depth_array,qpos_array,action_array,eef_qpos_array,episode_idx 

# parameters
step = 0
sub_step = 0
episode_idx = 0
Max_step = 400 #1000
Max_episode = 8
# directory_path = f'/media/dc/CLEAR/xgxy/dataset20241213' # f'/media/dc/ESD-USB/1120-remote-data'# f'/media/dc/HP2024/data/SCIL/Task4_long_horizon'

directory_path = f'/home/arxpro/ARX_Remote_Control/data/9_2_lemon_plate_2' # f'/media/dc/ESD-USB/1120-remote-data'# f'/media/dc/HP2024/data/SCIL/Task4_long_horizon'
extension = '.zarr' 
dataset_path = f'{directory_path}.zarr'
data_dict = {
        '/episode_ends' : [],
        '/qpos': [],
        '/action': [],
        '/eef_qpos': [],
        '/observations/images/mid' : [],
        '/observations/images/right' : [],
        '/observations/depth' : [],
        }
episode_ends_array,mid_image_array,right_image_array,depth_array,qpos_array,action_array,eef_qpos_array = [],[],[],[],[],[],[]

def callback(JointCTR2,JointInfo2,f2p,image_mid,image_right,depth):
    global data_dict, step, Max_step,Max_episode, dataset_path, episode_ends_array,mid_image_array,right_image_array,depth_array,qpos_array,action_array,eef_qpos_array,sub_step,episode_idx
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
        eef_qpos_array.append(eef_qpos)
        action_array.append(action)
        qpos_array.append(qpos)
        mid_image_array.append(image_mid)
        right_image_array.append(image_right)
        depth_array.append(depth)
        print(f"[DEBUG] Saved")

    canvas = np.zeros((480, 1280, 3), dtype=np.uint8)

    # 将图像复制到画布的特定位置
    # canvas[:, :640, :] = image_left
    # canvas[:, 640:1280, :] = image_mid
    # canvas[:, 1280:, :] = image_right
    canvas[:, :640, :] = image_mid
    canvas[:, 640:1280, :] = image_right

    # 在一个窗口中显示排列后的图像
    # cv2.imshow('Multi Camera Viewer', canvas)
  
    # cv2.waitKey(1)

    sub_step = sub_step + 1
    print(f"Step: {sub_step}")
    if sub_step >= Max_step and save:
        print(f'Episode {episode_idx+1} end__________________________________')
        step += sub_step
        sub_step = 0
        episode_idx = episode_idx + 1
        data_dict["/episode_ends"].append(step)
        data_dict["/eef_qpos"].extend(copy.deepcopy(eef_qpos_array))
        data_dict["/action"].extend(copy.deepcopy(action_array))
        data_dict["/qpos"].extend(copy.deepcopy(qpos_array))
        data_dict["/observations/images/mid"].extend(copy.deepcopy(mid_image_array))
        data_dict["/observations/images/right"].extend(copy.deepcopy(right_image_array))
        data_dict["/observations/depth"].extend(copy.deepcopy(depth_array))
        mid_image_array,right_image_array,depth_array,qpos_array,action_array,eef_qpos_array = [],[],[],[],[],[]
        print("Rest for 10 seconds")
        rospy.sleep(8)
        print("Wake up")

    if episode_idx >= Max_episode and save:
        # 保存数据到zarr文件
        zarr_root = zarr.group(dataset_path)
        zarr_data = zarr_root.create_group('data')
        zarr_meta = zarr_root.create_group('meta')
        # 转换数据格式
        data_dict['/observations/images/mid'] = np.stack(data_dict['/observations/images/mid'], axis=0)
        if data_dict['/observations/images/mid'].shape[1] == 3:  # 确保通道在最后
            data_dict['/observations/images/mid'] = np.transpose(data_dict['/observations/images/mid'], (0,2,3,1))
        data_dict['/observations/images/right'] = np.stack(data_dict['/observations/images/right'], axis=0)
        if data_dict['/observations/images/right'].shape[1] == 3:  # 确保通道在最后
            data_dict['/observations/images/right'] = np.transpose(data_dict['/observations/images/right'], (0,2,3,1))
        data_dict['/observations/depth'] = np.stack(data_dict['/observations/depth'], axis=0)
        data_dict['/eef_qpos'] = np.stack(data_dict['/eef_qpos'],axis=0)
        data_dict['/qpos'] = np.stack(data_dict['/qpos'],axis=0)
        data_dict['/action'] = np.stack(data_dict['/action'],axis=0)
        data_dict['/episode_ends'] = np.array(data_dict['/episode_ends'])

        # 设置压缩
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        
        # 设置chunk大小
        mid_img_chunk_size = (50, data_dict['/observations/images/mid'].shape[1], data_dict['/observations/images/mid'].shape[2], data_dict['/observations/images/mid'].shape[3])
        right_img_chunk_size = (50, data_dict['/observations/images/right'].shape[1], data_dict['/observations/images/right'].shape[2], data_dict['/observations/images/right'].shape[3])
        depth_chunk_size = (50, data_dict['/observations/depth'].shape[1], data_dict['/observations/depth'].shape[2])
        qpos_chunk_size = (50, data_dict['/qpos'].shape[1])
        eef_qpos_chunk_size = (50, data_dict['/eef_qpos'].shape[1])
        action_chunk_size = (50, data_dict['/action'].shape[1])

        
        # 创建数据集
        zarr_data.create_dataset('img_mid', data=data_dict['/observations/images/mid'], chunks=mid_img_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('img_right', data=data_dict['/observations/images/right'], chunks=right_img_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('depth', data=data_dict['/observations/depth'], chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('qpos', data=data_dict['/qpos'], chunks=qpos_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('eef_qpos', data=data_dict['/eef_qpos'], chunks=eef_qpos_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('action', data=data_dict['/action'], chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_meta.create_dataset('episode_ends',data=data_dict['/episode_ends'],dtype='int64',overwrite=True,compressor=compressor)
        # 打印数据信息
        cprint(f'-'*50, 'cyan')
        cprint(f'mid img shape: {data_dict["/observations/images/mid"].shape}, range: [{np.min(data_dict["/observations/images/mid"])}, {np.max(data_dict["/observations/images/mid"])}]', 'green')
        cprint(f'right img shape: {data_dict["/observations/images/right"].shape}, range: [{np.min(data_dict["/observations/images/right"])}, {np.max(data_dict["/observations/images/right"])}]', 'green') 
        cprint(f'depth shape: {data_dict["/observations/depth"].shape}, range: [{np.min(data_dict["/observations/depth"])}, {np.max(data_dict["/observations/depth"])}]', 'green')
        cprint(f'qpos shape: {data_dict["/qpos"].shape}, range: [{np.min(data_dict["/qpos"])}, {np.max(data_dict["/qpos"])}]', 'green')
        cprint(f'eef_qpos shape: {data_dict["/eef_qpos"].shape}, range: [{np.min(data_dict["/eef_qpos"])}, {np.max(data_dict["/eef_qpos"])}]', 'green')
        cprint(f'action shape: {data_dict["/action"].shape}, range: [{np.min(data_dict["/action"])}, {np.max(data_dict["/action"])}]', 'green')
        cprint(f'保存zarr文件到 {dataset_path}', 'green')
        
        # 清理内存
        del data_dict
        del zarr_root, zarr_data, zarr_meta
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
    
