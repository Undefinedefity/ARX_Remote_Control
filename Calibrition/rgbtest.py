import cv2
import numpy as np
import os

# ======================
# 1. 标定RGB相机内参（修改版）
# ======================

def calibrate_rgb_camera(image_folder, pattern_size=(7, 7), square_size=15):
    """
    使用棋盘格标定RGB相机内参
    :return: 内参矩阵, 畸变系数, 平均误差, 对象点, 图像点
    """
    # 准备对象点（3D坐标）
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []  # 3D点
    imgpoints = []  # 2D点

    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    mean_error = compute_reprojection_error(objpoints, imgpoints, mtx, dist)
    
    return mtx, dist, mean_error, objpoints, imgpoints

# ======================
# 2. 重投影误差计算函数（新增）
# ======================

def compute_reprojection_error(objpoints, imgpoints, mtx, dist):
    """
    计算任意内参的重投影误差
    """
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        # 计算每个图像的姿态
        ret, rvec, tvec = cv2.solvePnP(objpoints[i], imgpoints[i], mtx, dist)
        if not ret:
            continue
            
        # 投影3D点到图像平面
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvec, tvec, mtx, dist)
        
        # 计算L2误差
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        total_error += error ** 2
        total_points += len(imgpoints[i])
    
    return np.sqrt(total_error / total_points) if total_points > 0 else float('inf')

# ======================
# 3. 参数验证函数（增强版）
# ======================

def validate_rgb_calibration(mtx, dist, image_size=(640, 480)):
    """
    增强版参数验证，包含主点检查
    """
    # 理论FOV参数
    rgb_h_fov = 69  # 水平方向FOV（度）
    rgb_v_fov = 42  # 垂直方向FOV（度）
    
    # 计算理论值
    expected_fx = image_size[0] / (2 * np.tan(np.deg2rad(rgb_h_fov) / 2))
    expected_fy = image_size[1] / (2 * np.tan(np.deg2rad(rgb_v_fov) / 2))
    expected_cx = image_size[0] / 2
    expected_cy = image_size[1] / 2

    # 实际参数
    actual_fx = mtx[0, 0]
    actual_fy = mtx[1, 1]
    actual_cx = mtx[0, 2]
    actual_cy = mtx[1, 2]

    print("\n验证结果：")
    print(f"理论焦距 | fx: {expected_fx:.2f}  fy: {expected_fy:.2f}")
    print(f"实际焦距 | fx: {actual_fx:.2f}  fy: {actual_fy:.2f}")
    print(f"主点坐标 | 理论中心 ({expected_cx}, {expected_cy})，实际 ({actual_cx:.1f}, {actual_cy:.1f})")

    # 误差检查
    validation = {
        '焦距_X差异%': abs((actual_fx - expected_fx)/expected_fx)*100,
        '焦距_Y差异%': abs((actual_fy - expected_fy)/expected_fy)*100,
        '主点X偏移': abs(actual_cx - expected_cx),
        '主点Y偏移': abs(actual_cy - expected_cy)
    }

    print("\n验收标准检查：")
    print(f"1. 焦距差异（应<15%）: X轴 {validation['焦距_X差异%']:.1f}%，Y轴 {validation['焦距_Y差异%']:.1f}%")
    print(f"2. 主点偏移（应<30像素）: X偏移 {validation['主点X偏移']:.1f}，Y偏移 {validation['主点Y偏移']:.1f}")
    print(f"3. 畸变系数范围（应<0.5）: {dist.flatten().round(3)}")

    # 返回验证状态
    passed = (
        validation['焦距_X差异%'] < 15 and
        validation['焦距_Y差异%'] < 15 and
        validation['主点X偏移'] < 30 and
        validation['主点Y偏移'] < 30 and
        all(abs(dist.flatten()) < 0.5)
    )
    print("\n综合验证结果：", "通过" if passed else "未通过")
    return passed
    
def get_chessboard_points(image_folder, pattern_size=(7,7), square_size=15):
    """
    从指定文件夹获取棋盘格数据点
    :return: 对象点列表, 图像点列表
    """
    # 准备3D对象点（与实际标定时一致）
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []

    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
              if f.endswith(('.jpg', '.png'))]
    
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    return objpoints, imgpoints
# ======================
# 4. 主程序（比较版）
# ======================

if __name__ == "__main__":
    # 标定使用 test_images 数据集
    rgb_image_folder = "calibrition/test_images"
    new_mtx, new_dist, new_error, _, _ = calibrate_rgb_camera(rgb_image_folder)
    
    # 现有参数配置（保持不变）
    existing_mtx = np.array([
        [617.0569458007812, 0, 318.4468688964844],
        [0, 617.5814819335938, 245.88998413085938],
        [0, 0, 1]
    ], dtype=np.float32)
    existing_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # 从 test_images1 获取测试数据
    test_folder = "calibrition/test_images1"
    objpoints_test, imgpoints_test = get_chessboard_points(test_folder)

    # 计算新参数在测试集的误差
    new_test_error = compute_reprojection_error(objpoints_test, imgpoints_test, new_mtx, new_dist)
    
    # 计算现有参数在测试集的误差
    existing_test_error = compute_reprojection_error(objpoints_test, imgpoints_test, existing_mtx, existing_dist)

    # 打印对比结果
    print("\n======== 测试集对比结果 ========")
    print(f"| 参数类型 | 标定误差 | 测试集误差 |")
    print(f"| 新参数 | {new_error:.4f} | {new_test_error:.4f} |")
    print(f"| 现有参数 | - | {existing_test_error:.4f} |")
    print("注：测试集误差更能反映泛化能力")

    # 参数验证（仍使用标定数据）
    print("\n==== 新标定参数验证 ====")
    new_valid = validate_rgb_calibration(new_mtx, new_dist)
    
    print("\n==== 现有参数验证 ====")
    existing_valid = validate_rgb_calibration(existing_mtx, existing_dist)

    # 最终建议
    print("\n======== 最终建议 ========")
    criteria_passed = {
        'new_valid': new_valid,
        'existing_valid': existing_valid,
        'new_test_better': new_test_error < existing_test_error
    }
    
    if criteria_passed['new_valid'] and criteria_passed['new_test_better']:
        print("强烈建议采用新参数，因为：")
        print("- 通过所有验证检查")
        print(f"- 测试集误差更低 ({new_test_error:.4f} vs {existing_test_error:.4f})")
    elif criteria_passed['existing_valid'] and not criteria_passed['new_valid']:
        print("建议保留现有参数，因为：")
        print("- 新参数验证未通过")
        print("- 现有参数仍然有效")
    else:
        print("建议重新校准，因为：")
        if not criteria_passed['new_valid']:
            print("- 新参数验证失败")
        if not criteria_passed['existing_valid']:
            print("- 现有参数验证失败")
        if not criteria_passed['new_test_better']:
            print(f"- 新参数在测试集表现不佳 ({new_test_error:.4f} vs {existing_test_error:.4f})")
