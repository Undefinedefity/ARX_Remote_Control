import cv2
import numpy as np

# 定义标定板参数
square_size = 15  # 棋盘格每格边长（毫米）
pattern_size = (7, 7)  # 内角点数量（行,列）
obj_points = np.zeros((pattern_size[0]*pattern_size[1], 3), dtype=np.float32)
obj_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# 读取图像
img = cv2.imread("calib_images1/chessboard_004.png")
if img is None:
    raise FileNotFoundError("无法加载图像，请检查路径和文件名！")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测棋盘格角点
ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

if ret:
    # 创建新图像用于可视化
    img_with_points = img.copy()
    
    # 获取原点（左上角）和最右下角点的坐标
    origin = tuple(corners[0][0].astype(int))        # 绿色
    bottom_right = tuple(corners[-1][0].astype(int))  # 红色
    
    # 绘制原点（绿色十字）
    cv2.line(img_with_points, (origin[0]-10, origin[1]), (origin[0]+10, origin[1]), (0, 255, 0), 2)
    cv2.line(img_with_points, (origin[0], origin[1]-10), (origin[0], origin[1]+10), (0, 255, 0), 2)
    
    # 绘制最右下角点（红色十字）
    cv2.line(img_with_points, (bottom_right[0]-10, bottom_right[1]), (bottom_right[0]+10, bottom_right[1]), (0, 0, 255), 2)
    cv2.line(img_with_points, (bottom_right[0], bottom_right[1]-10), (bottom_right[0], bottom_right[1]+10), (0, 0, 255), 2)
    
    # 显示结果
    cv2.imshow("Detected Key Points", img_with_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 已知内参和畸变系数
    fx = 617.0569458007812
    fy = 617.5814819335938
    cx = 318.4468688964844
    cy = 245.88998413085938
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # 假设无畸变

    # 直接使用原始角点及内参求解外参（因D为零，无需去畸变）
    success, rvec, tvec = cv2.solvePnP(obj_points, corners, K, D)
    

    # 输出外参
    R, _ = cv2.Rodrigues(rvec)
    print("\n外参旋转矩阵 R:")
    print(R)
    print("\n外参平移向量 t (毫米):")
    print(tvec)
    
    # ============ 新增部分：输出原点在相机坐标系的坐标 ============
    # 标定板原点（棋盘格左上角）在相机坐标系中的坐标即为平移向量 t
    origin_in_camera = tvec.flatten()  # 将3x1向量转为1D数组
    print("\n标定板原点在相机坐标系中的坐标 (毫米):")
    print(f"X: {origin_in_camera[0]:.2f} mm")
    print(f"Y: {origin_in_camera[1]:.2f} mm")
    print(f"Z: {origin_in_camera[2]:.2f} mm")
    # ========================================================
    
    # 计算重投影误差
    projected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)
    error = cv2.norm(corners, projected_points, cv2.NORM_L2) / len(projected_points)
    print("平均重投影误差 (像素):", error)
