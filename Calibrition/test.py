import cv2
import numpy as np
import os

# ======================
# 1. 标定RGB相机内参
# ======================

def calibrate_rgb_camera(image_folder, pattern_size=(7, 7), square_size=15):
    """
    使用棋盘格标定RGB相机内参
    :param image_folder: 包含标定图像的文件夹路径
    :param pattern_size: 棋盘格内角点数量（行,列）
    :param square_size: 棋盘格方格实际尺寸（毫米）
    :return: 相机内参矩阵、畸变系数、平均重投影误差
    """
    # 准备对象点（3D坐标）
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    # 存储所有图像的对象点和图像点
    objpoints = []  # 3D点
    imgpoints = []  # 2D点

    # 遍历标定图像
    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # 可视化检测结果
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # 标定相机
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # 计算平均重投影误差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)

    return mtx, dist, mean_error


# ======================
# 2. 验证深度相机内参（更新版）
# ======================

def validate_depth_camera(depth_intrin, expected_size=(640, 480)):
    """
    验证深度相机内参是否符合规格
    :param depth_intrin: 包含相机参数的字典（必须包含fx, fy, cx, cy）
    :param expected_size: 预期的图像分辨率（宽, 高）
    :return: 验证是否通过（bool）
    """
    # 官网提供的深度相机FOV参数
    depth_h_fov = 87  # 水平方向FOV（度）
    depth_v_fov = 58  # 垂直方向FOV（度）
    
    # 计算理论焦距
    expected_fx = expected_size[0] / (2 * np.tan(np.deg2rad(depth_h_fov) / 2))
    expected_fy = expected_size[1] / (2 * np.tan(np.deg2rad(depth_v_fov) / 2))
    
    expected_cx = expected_size[0] / 2
    expected_cy = expected_size[1] / 2

    print("\n深度相机理论参数：")
    print(f"fx: {expected_fx:.2f}  fy: {expected_fy:.2f}")
    print(f"cx: {expected_cx}  cy: {expected_cy}")
    
    print("\n实际相机参数：")
    print(f"fx: {depth_intrin['fx']:.2f}  fy: {depth_intrin['fy']:.2f}")
    print(f"cx: {depth_intrin['cx']:.2f}  cy: {depth_intrin['cy']:.2f}")

    # 验证参数容差
    validation_passed = True
    # 焦距验证（允许10%误差）
    if not np.isclose(depth_intrin['fx'], expected_fx, rtol=0.1):
        print(f"警告：X轴焦距偏差超过10%！理论值 {expected_fx:.2f}，实际 {depth_intrin['fx']:.2f}")
        validation_passed = False
    if not np.isclose(depth_intrin['fy'], expected_fy, rtol=0.1):
        print(f"警告：Y轴焦距偏差超过10%！理论值 {expected_fy:.2f}，实际 {depth_intrin['fy']:.2f}")
        validation_passed = False
    
    # 主点验证（允许20像素偏移）
    if not np.isclose(depth_intrin['cx'], expected_cx, atol=20):
        print(f"警告：X轴主点偏移超过20像素！理论值 {expected_cx}，实际 {depth_intrin['cx']}")
        validation_passed = False
    if not np.isclose(depth_intrin['cy'], expected_cy, atol=20):
        print(f"警告：Y轴主点偏移超过20像素！理论值 {expected_cy}，实际 {depth_intrin['cy']}")
        validation_passed = False

    print("\n验证结果：", "通过" if validation_passed else "未通过")
    return validation_passed


# ======================
# 3. RGB相机参数验证
# ======================

def validate_rgb_calibration(rgb_mtx, image_size=(640, 480)):
    """
    验证RGB标定结果合理性
    :param rgb_mtx: 标定得到的内参矩阵
    :param image_size: 图像分辨率（宽, 高）
    """
    # 官网提供的RGB相机FOV参数
    rgb_h_fov = 69  # 水平方向FOV（度）
    rgb_v_fov = 42  # 垂直方向FOV（度）
    
    # 计算理论焦距
    expected_fx = image_size[0] / (2 * np.tan(np.deg2rad(rgb_h_fov) / 2))
    expected_fy = image_size[1] / (2 * np.tan(np.deg2rad(rgb_v_fov) / 2))
    
    print("\nRGB相机理论焦距：")
    print(f"fx: {expected_fx:.2f}  fy: {expected_fy:.2f}")
    print("实际标定结果：")
    print(f"fx: {rgb_mtx[0,0]:.2f}  fy: {rgb_mtx[1,1]:.2f}")
    
    # 建议误差范围
    print("\n建议验收标准：")
    print("1. 重投影误差 < 0.5像素")
    print("2. 实际焦距与理论值差异 < 15%")
    print("3. 主点坐标应在图像中心 ±30像素范围内")


# ======================
# 4. 主程序
# ======================

if __name__ == "__main__":
    # 标定RGB相机
    rgb_image_folder = "calibrition/test_images"
    rgb_mtx, rgb_dist, rgb_error = calibrate_rgb_camera(rgb_image_folder)
    
    print("\nRGB相机标定结果：")
    print(f"内参矩阵：\n{rgb_mtx}")
    print(f"畸变系数：{rgb_dist.flatten()}")
    print(f"平均重投影误差：{rgb_error:.4f} 像素")
    
    # 验证RGB标定结果合理性
    validate_rgb_calibration(rgb_mtx, image_size=(640, 480))

    # 验证深度相机参数（示例数据）
    depth_intrin = {
        'fx': 385.98,
        'fy': 385.98,
        'cx': 323.75,
        'cy': 238.28,
        'width': 640,
        'height': 480
    }
    print("\n开始深度相机参数验证...")
    depth_valid = validate_depth_camera(depth_intrin)
    print("深度相机验证最终结果：", "通过" if depth_valid else "需要重新校准")
