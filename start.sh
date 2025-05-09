workspace=$(pwd)

# 启动第一个 follow1 文件
gnome-terminal -- bash -c "cd ${workspace}/follow1 && source devel/setup.bash && roslaunch arm_control arx5v.launch"

# # 启动第一个 follow2 文件
# gnome-terminal -- bash -c "cd /home/dc/Desktop/arx-follow-V2/arx-follow/remote_control/follow2 && source devel/setup.bash && roslaunch arm_control arx5v.launch"
sleep 0.5
# # 启动第一个 master1 文件
gnome-terminal -- bash -c "cd ${workspace}/master1 && source devel/setup.bash && roslaunch arm_control arx5v.launch"

# 启动第一个 master2 文件
# gnome-terminal -- bash -c "cd /home/dc/Desktop/arx-follow-V2/arx-follow/remote_control/master2 && source devel/setup.bash && roslaunch arm_control arx5v.launch"


sleep 0.5
gnome-terminal -- bash -c "cd ${workspace}/follow1 && source devel/setup.bash && roslaunch arm_control camera.launch"