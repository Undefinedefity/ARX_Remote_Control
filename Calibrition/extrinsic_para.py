import cv2
import numpy as np

# 定义标定板参数
square_size = 15  # 棋盘格每格边长（毫米）
pattern_size = (7, 7)  # 内角点数量（行,列）
obj_points = np.zeros((pattern_size[0]*pattern_size[1], 3), dtype=np.float32)
obj_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# 读取图像
img = cv2.imread("Calibrition/im_Color.png")
if img is None:
    raise FileNotFoundError("无法加载图像，请检查路径和文件名！")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测棋盘格角点
ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

if ret:
    # 可视化角点
    img_with_corners = img.copy()
    cv2.drawChessboardCorners(img_with_corners, pattern_size, corners, ret)
    cv2.imshow("Detected Chessboard Corners", img_with_corners)
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
    
    # 计算重投影误差
    projected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)
    error = cv2.norm(corners, projected_points, cv2.NORM_L2) / len(projected_points)
    print("平均重投影误差 (像素):", error)
else:
    print("棋盘格角点检测失败！请检查：")
    print("- 图像中棋盘格是否完整可见")
    print(f"- pattern_size 参数设置（当前为 {pattern_size} 个内角点）")