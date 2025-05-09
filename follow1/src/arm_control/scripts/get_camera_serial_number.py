import pyrealsense2 as rs

# 创建一个上下文对象，获取所有连接的设备
context = rs.context()

# 遍历所有设备
for device in context.devices:
    # 获取每个设备的序列号
    serial_number = device.get_info(rs.camera_info.serial_number)
    print(f"Device serial number: {serial_number}")
