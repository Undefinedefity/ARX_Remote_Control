#!/home/dc/anaconda3/envs/dc/bin/python
import time
import rospy
import sys
from message_filters import ApproximateTimeSynchronizer,Subscriber
sys.path.append("/home/dc/anaconda3/envs/dc/lib/python3.8/site-packages")
import numpy as np
import cv2
import h5py
from cv_bridge import CvBridge
from arm_control.msg import JointInformation
from arm_control.msg import JointControl
from arm_control.msg import PosCmd
from sensor_msgs.msg import Image
import os
from pynput import keyboard

# intr white cam
# camera_matrix = np.array([[607.143676757812, 0, 318.693939208984],
#                           [0, 606.600341796875, 252.772720336914],
#                           [0, 0, 1]], dtype=float)

# intr black cam
camera_matrix = np.array([[604.609191894531, 0, 319.987457275391],
                          [0, 604.114990234375, 257.497772216797],
                          [0, 0, 1]], dtype=float)



# ignore for now
dist_coeffs = np.array([0, 0, 0, 0], dtype=float)

def on_press(key):
    global save_image_flag
    try:
        if key == keyboard.Key.enter:
            save_image_flag = True
    except AttributeError:
        pass

def count_files_with_extension(directory, extension):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count

def get_marker_pose(corners, ids, board):
    retval, rvec, tvec = cv2.aruco.estimatePoseBoard(corners=corners,
                                                     ids=ids,
                                                     board=board,
                                                     cameraMatrix=camera_matrix,
                                                     distCoeffs=dist_coeffs, rvec=None, tvec=None)
    # print("rvec: ", rvec)
    # print("tvec: ", tvec)
    pose = np.concatenate((tvec, rvec), axis=None)
    return pose


global data_dict, step, Max_step, dataset_path, save_image_flag

# parameters
step = 0
Max_step = 20
directory_path = f'/media/dc/ESD-USB/calib_test'# f'/media/dc/HP2024/data/SCIL/Task4_long_horizon'
extension = '.hdf5' 
episode_idx = count_files_with_extension(directory_path, extension)
dataset_path = f'{directory_path}/episode_{episode_idx}.hdf5'
video_path= directory_path + f'/video/{episode_idx}'
data_dict = {
        '/observations/qpos': [],
        '/observations/action': [],
        '/observations/eef_qpos': [],
        '/observations/images/mid' : [],
        '/observations/marker_poses' : [],
        # '/observations/images/right' : [],
        }
save_image_flag = False

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters =  cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
# metal board
# num, size(width), gap, [6*6], ids (default from 0)
board = cv2.aruco.GridBoard((4, 3), 0.038, 0.0076, aruco_dict,
                            ids=np.array([36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]))
# board = cv2.aruco.GridBoard((1, 1), 0.0995, 0.01, aruco_dict)

# def callback(JointCTR2,JointInfo2,f2p,image_mid,image_right):
def callback(JointCTR2,JointInfo2,f2p,image_mid):
    global data_dict, step, Max_step, dataset_path, video_path, save_image_flag
    
    save=True
    bridge = CvBridge()
    image_mid = bridge.imgmsg_to_cv2(image_mid, "bgr8")
    # image_right = bridge.imgmsg_to_cv2(image_right, "bgr8")
    eef_qpos=np.array([f2p.x,f2p.y,f2p.z,f2p.roll,f2p.pitch,f2p.yaw,f2p.gripper])
    action = np.array(JointCTR2.joint_pos)
    qpos =np.array(JointInfo2.joint_pos)
    gray = cv2.cvtColor(image_mid, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected_img_points = aruco_detector.detectMarkers(gray)
    detected_corners, detected_ids, rejected_corners, recovered_ids = aruco_detector.refineDetectedMarkers(
        detectedCorners=corners,
        detectedIds=ids,
        rejectedCorners=rejected_img_points,
        image=gray,
        board=board,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
    )

    if len(data_dict["/observations/qpos"])>0 and np.linalg.norm(qpos-data_dict["/observations/qpos"][-1]) <= 0.05:
        return
    # print("eef_qpos:", eef_qpos)
    # print("action:", action)
    if save and save_image_flag:
        if len(detected_corners) > 0:
            data_dict["/observations/eef_qpos"].append(eef_qpos)
            data_dict["/observations/action"].append(action)
            data_dict["/observations/qpos"].append(qpos)
            data_dict["/observations/images/mid"].append(image_mid)
            marker_pose = get_marker_pose(detected_corners, detected_ids, board)
            print("marker_pose: ", marker_pose)
            data_dict["/observations/marker_poses"].append(marker_pose)
            save_image_flag = False
            step = step + 1
            cv2.aruco.drawDetectedMarkers(image_mid, corners, ids)
            print("Saved one frame data!")
        else:
            print("Aruco not detected")
        # data_dict["/observations/images/right"].append(image_right)

    canvas = np.zeros((480, 1280, 3), dtype=np.uint8)

    # 将图像复制到画布的特定位置
    # canvas[:, :640, :] = image_left
    # canvas[:, 640:1280, :] = image_mid
    # canvas[:, 1280:, :] = image_right
    canvas[:, :640, :] = image_mid
    # canvas[:, 640:1280, :] = image_right

    # 在一个窗口中显示排列后的图像
    cv2.imshow('Multi Camera Viewer', canvas)
  
    cv2.waitKey(1)

    print(step)
    if step >= Max_step and save:
        print('end__________________________________')
        with h5py.File(dataset_path,'w',rdcc_nbytes=1024 ** 2 * 10) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            _ = image.create_dataset('mid', (Max_step, 480, 640, 3), dtype='uint8',
                                    chunks=(1, 480, 640, 3), )
            # _ = image.create_dataset('right', (Max_step, 480, 640, 3), dtype='uint8',
                                    # chunks=(1, 480, 640, 3), )
            _ = obs.create_dataset('qpos',(Max_step,7))
            _ = obs.create_dataset('action',(Max_step,7))
            _ = obs.create_dataset('eef_qpos',(Max_step,7))
            _ = obs.create_dataset('marker_poses',(Max_step,6))
            for name, array in data_dict.items():
                root[name][...] = array
            mid_images = root['/observations/images/mid'][...]
            # right_images = root['/observations/images/right'][...]
            # images = np.concatenate([mid_images,right_images],axis=2)

            video_path = f'{video_path}video.mp4'  # Assuming dataset_path ends with ".hdf5"
            height, width, _ = mid_images[0].shape
            fps = 10  # 发布频率为10Hz
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for img in mid_images:
                video_writer.write(img)
            video_writer.release()
        rospy.signal_shutdown("\n************************signal_shutdown********sample successfully!*************************************")
        quit("sample successfully!")
        

if __name__ =="__main__":
    #config my camera
    time.sleep(2)  # wait 2s to start
    
    rospy.init_node("sample_ik")
    
    a=time.time()
    # master1_pos = Subscriber("master1_pos_back",PosCmd)
    # master2_pos = Subscriber("master2_pos_back",PosCmd)
    follow1_pos = Subscriber("follow1_pos_back",PosCmd)
    # follow2_pos = Subscriber("follow2_pos_back",PosCmd)
    master1 = Subscriber("joint_control",JointControl)
    # master2 = Subscriber("joint_control2",JointControl)
    follow1 = Subscriber("joint_information",JointInformation)
    # follow2 = Subscriber("joint_information2",JointInformation)
    # image_mid = Subscriber("mid_camera",Image)
    image_left = Subscriber("left_camera",Image)
    # image_right = Subscriber("right_camera",Image)
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    # ats = ApproximateTimeSynchronizer([master1,follow1,follow1_pos,image_mid],slop=0.003,queue_size=2)
    ats = ApproximateTimeSynchronizer([master1,follow1,follow1_pos,image_left],slop=0.03,queue_size=2)
    ats.registerCallback(callback)
    rospy.spin()
    
