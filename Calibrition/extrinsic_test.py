import os
import cv2
import numpy as np
import csv

# 配置参数
IMAGE_DIR = "calibrition/calib_images1"  # 包含5张标定图像的文件夹
PATTERN_SIZE = (7, 7)
SQUARE_SIZE = 15

# 已知的相机到机械臂基座标系变换矩阵
CAM_TO_ROBOT = np.array([
    [ 0.75422983,  0.23041472, -0.6148548,  468.233   ],
    [-0.65643515,  0.28624051, -0.69796795, 1090.323  ],
    [ 0.01517426,  0.93004055,  0.36714346, -472.788  ],
    [ 0.0,         0.0,         0.0,         1.0      ]
])

# 预定义内参矩阵和畸变系数
FX = 617.0569458007812
FY = 617.5814819335938
CX = 318.4468688964844
CY = 245.88998413085938
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# 图像文件名到测量坐标的映射字典
MEASURED_COORDS = {
    "image1.jpg": np.array([100.0, 200.0, 50.0]),  # 替换为实际测量值
    "chessboard_001.png": np.array([175, 400, 126),  # 替换为实际测量值
    "chessboard_002.png": np.array([100, 275, 126]),  # 替换为实际测量值
    "chessboard_003.png": np.array([-12, 560, 3]),  # 替换为实际测量值
    "chessboard_004.png": np.array([75, 510, 3]),  # 替换为实际测量值
}

# 生成3D标定板坐标点
obj_points = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1], 3), dtype=np.float32)
obj_points[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

def transform_to_robot_frame(camera_coord):
    """将相机坐标系坐标转换到机械臂基座标系"""
    homogeneous_coord = np.append(camera_coord, 1.0)
    robot_coord = CAM_TO_ROBOT @ homogeneous_coord
    return robot_coord[:3]

def calculate_errors(calculated, measured):
    """计算误差统计信息"""
    errors = [np.linalg.norm(c - m) for c, m in zip(calculated, measured)]
    return {
        'average': np.mean(errors),
        'max': np.max(errors),
        'min': np.min(errors),
        'std': np.std(errors),
        'all': errors
    }

def process_existing_images(image_dir):
    """处理已有图像并计算坐标"""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    calculated_points = []
    measured_points = []
    
    print(f"\n找到 {len(image_files)} 张图像，开始处理...")
    
    for idx, filename in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图像 {filename}，已跳过")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)
        
        if not ret:
            print(f"图像 {filename} 未检测到棋盘格，已跳过")
            continue
            
        # 优化角点检测
        corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        # 计算外参
        success, rvec, tvec = cv2.solvePnP(obj_points, corners_refined, K, D)
        if not success:
            print(f"图像 {filename} 外参计算失败，已跳过")
            continue
        
        # 坐标转换
        origin_camera = tvec.flatten()
        origin_robot = transform_to_robot_frame(origin_camera)
        
        # 获取测量坐标
        if filename in MEASURED_COORDS:
            measured = MEASURED_COORDS[filename]
        else:
            print(f"警告：图像 {filename} 没有对应的测量坐标，已跳过")
            continue
        
        # 显示结果
        print(f"\n=== 图像 {idx}/{len(image_files)} ===")
        print(f"文件: {filename}")
        print("计算坐标 (机械臂坐标系):")
        print(f"X: {origin_robot[0]:.2f}mm  Y: {origin_robot[1]:.2f}mm  Z: {origin_robot[2]:.2f}mm")
        print("测量坐标 (机械臂坐标系):")
        print(f"X: {measured[0]:.2f}mm  Y: {measured[1]:.2f}mm  Z: {measured[2]:.2f}mm")
        
        # 保存数据
        calculated_points.append(origin_robot)
        measured_points.append(measured)
        
        # 显示当前误差
        current_error = np.linalg.norm(origin_robot - measured)
        print(f"当前误差: {current_error:.2f}mm")
        print("="*40)
    
    # 计算最终统计
    if calculated_points:
        stats = calculate_errors(calculated_points, measured_points)
        
        print("\n=== 最终误差统计 ===")
        print(f"平均误差: {stats['average']:.2f}mm")
        print(f"最大误差: {stats['max']:.2f}mm")
        print(f"最小误差: {stats['min']:.2f}mm")
        print(f"标准差: {stats['std']:.2f}mm")
        
        # 保存到CSV
        csv_path = os.path.join(image_dir, "calibration_report.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                '图像文件', 
                '计算_X', '计算_Y', '计算_Z',
                '实测_X', '实测_Y', '实测_Z',
                '误差(mm)'
            ])
            for fname, c, m, e in zip(image_files, calculated_points, measured_points, stats['all']):
                writer.writerow([
                    fname,
                    f"{c[0]:.2f}", f"{c[1]:.2f}", f"{c[2]:.2f}",
                    f"{m[0]:.2f}", f"{m[1]:.2f}", f"{m[2]:.2f}",
                    f"{e:.2f}"
                ])
        print(f"\n详细报告已保存至: {csv_path}")
    else:
        print("警告：未成功处理任何有效图像")

if __name__ == "__main__":
    print("标定板坐标验证程序")
    print("操作流程：")
    print("1. 程序自动处理指定目录下的图像")
    print("2. 对每张有效图像显示计算结果和预设测量值")
    print("3. 自动计算误差")
    print("4. 最终生成误差统计报告\n")
    
    process_existing_images(IMAGE_DIR)
