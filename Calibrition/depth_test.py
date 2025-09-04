import cv2
import numpy as np
import os
import json
import random

# ======================
# 通用标定函数
# ======================

def calibrate_camera(image_folder, pattern_size=(7,7), square_size=15, validation_split=0.2):
    """
    通用相机标定函数（适用于深度/RGB相机）
    返回：内参矩阵, 畸变系数, 训练集误差, 验证集误差, 验证集对象点, 验证集图像点
    """
    # 准备3D对象点
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []  # 3D点集合
    imgpoints = []  # 2D点集合

    # 读取所有标定图像
    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(images) == 0:
        raise ValueError("未找到标定图像！请检查目录路径")

    valid_count = 0
    for idx, fname in enumerate(images):
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)  # IR图像为灰度图
        if img is None:
            continue

        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(img, pattern_size, 
                                                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                      cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            # 亚像素精确化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            valid_count += 1

    if valid_count < 10:
        raise ValueError(f"有效图像不足（{valid_count}张），至少需要10张有效图像")

    # ===== 随机分割训练集和验证集 =====
    indices = list(range(len(objpoints)))
    random.shuffle(indices)
    split_idx = int(len(indices) * (1 - validation_split))
    
    train_indices = indices[:split_idx]
    valid_indices = indices[split_idx:]
    
    train_obj = [objpoints[i] for i in train_indices]
    train_img = [imgpoints[i] for i in train_indices]
    valid_obj = [objpoints[i] for i in valid_indices]
    valid_img = [imgpoints[i] for i in valid_indices]

    # 使用训练集标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        train_obj, train_img, img.shape[::-1], None, None,
        flags=cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST
    )

    # 计算训练集重投影误差
    train_error = 0
    for i in range(len(train_obj)):
        imgpoints2, _ = cv2.projectPoints(train_obj[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(train_img[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        train_error += error
    train_error /= len(train_obj)

    # 计算验证集重投影误差
    valid_error = 0
    valid_count = 0
    for i in range(len(valid_obj)):
        ret, rvec, tvec = cv2.solvePnP(valid_obj[i], valid_img[i], mtx, dist)
        if ret:
            projected, _ = cv2.projectPoints(valid_obj[i], rvec, tvec, mtx, dist)
            error = cv2.norm(valid_img[i], projected, cv2.NORM_L2) / len(projected)
            valid_error += error
            valid_count += 1
    valid_error /= valid_count if valid_count > 0 else 1

    return mtx, dist, train_error, valid_error, valid_obj, valid_img

# ======================
# 参数验证函数
# ======================

def validate_depth_camera(depth_intrin, expected_size=(640, 480)):
    """
    验证深度相机内参是否符合规格
    :param depth_intrin: 包含相机参数的字典（必须包含fx, fy, cx, cy）
    :param expected_size: 预期的图像分辨率（宽, 高）
    :return: 验证是否通过（bool）
    """
    # 官网提供的深度相机FOV参数（以Intel D435i为例）
    depth_h_fov = 87  # 水平方向FOV（度）
    depth_v_fov = 58  # 垂直方向FOV（度）
    
    # ===== 1. 计算理论参数 =====
    # 理论焦距计算公式：f = (width / 2) / tan(FOV/2)
    expected_fx = expected_size[0] / (2 * np.tan(np.deg2rad(depth_h_fov) / 2))
    expected_fy = expected_size[1] / (2 * np.tan(np.deg2rad(depth_v_fov) / 2))
    
    # 理论主点应为图像中心
    expected_cx = expected_size[0] / 2
    expected_cy = expected_size[1] / 2

    # ===== 2. 打印对比信息 =====
    print("\n深度相机理论参数：")
    print(f"fx: {expected_fx:.2f}  fy: {expected_fy:.2f}")
    print(f"cx: {expected_cx:.1f}  cy: {expected_cy:.1f}")
    
    print("\n实际相机参数：")
    print(f"fx: {depth_intrin['fx']:.2f}  fy: {depth_intrin['fy']:.2f}")
    print(f"cx: {depth_intrin['cx']:.2f}  cy: {depth_intrin['cy']:.2f}")

    # ===== 3. 参数验证逻辑 =====
    validation_passed = True
    
    # 焦距验证（允许10%误差）
    if not np.isclose(depth_intrin['fx'], expected_fx, rtol=0.1):
        print(f"⚠️ X轴焦距偏差超过10%！理论值 {expected_fx:.2f}，实际 {depth_intrin['fx']:.2f}")
        validation_passed = False
    
    if not np.isclose(depth_intrin['fy'], expected_fy, rtol=0.1):
        print(f"⚠️ Y轴焦距偏差超过10%！理论值 {expected_fy:.2f}，实际 {depth_intrin['fy']:.2f}")
        validation_passed = False
    
    # 主点验证（允许20像素偏移）
    if not np.isclose(depth_intrin['cx'], expected_cx, atol=20):
        print(f"⚠️ X主点偏移超过20像素！理论值 {expected_cx:.1f}，实际 {depth_intrin['cx']:.2f}")
        validation_passed = False
    
    if not np.isclose(depth_intrin['cy'], expected_cy, atol=20):
        print(f"⚠️ Y主点偏移超过20像素！理论值 {expected_cy:.1f}，实际 {depth_intrin['cy']:.2f}")
        validation_passed = False

    # ===== 4. 输出最终结果 =====
    print("\n验证结果：", "✅ 通过" if validation_passed else "❌ 未通过")
    return validation_passed

# ======================

def main():
    # 配置文件路径
    CALIB_IMAGES_DIR = "calibrition/ir_images1"  # 替换为实际路径
    ORIGINAL_INTRIN_FILE = "original_depth_intrin.json"  # 原始参数保存文件
    
    # 原始深度相机参数（示例结构）
    original_intrin = {
        "fx": 385.98,
        "fy": 385.98,
        "cx": 323.75,
        "cy": 238.28,
        "width": 640,
        "height": 480
    }
    
    # Step 1: 标定新参数
    try:
        new_mtx, new_dist, train_error, valid_error, valid_obj, valid_img = calibrate_camera(
            image_folder=CALIB_IMAGES_DIR,
            pattern_size=(7,7),  # 根据实际棋盘格调整
            square_size=15        # 根据实际棋盘格尺寸调整（毫米）
        )
    except Exception as e:
        print(f"标定失败：{str(e)}")
        return

    # Step 2: 构造新参数字典
    new_intrin = {
        "fx": new_mtx[0,0],
        "fy": new_mtx[1,1],
        "cx": new_mtx[0,2],
        "cy": new_mtx[1,2],
        "width": 640,   # 根据实际分辨率调整
        "height": 480
    }

    # Step 3: 计算原始参数在验证集上的误差
    # 构造原始参数矩阵（假设无畸变）
    original_mtx = np.array([
        [original_intrin["fx"], 0, original_intrin["cx"]],
        [0, original_intrin["fy"], original_intrin["cy"]],
        [0, 0, 1]
    ], dtype=np.float32)
    original_dist = np.zeros(5)  # 假设原始参数无畸变

    original_valid_error = 0
    valid_count = 0
    for i in range(len(valid_obj)):
        ret, rvec, tvec = cv2.solvePnP(valid_obj[i], valid_img[i], original_mtx, original_dist)
        if ret:
            projected, _ = cv2.projectPoints(valid_obj[i], rvec, tvec, original_mtx, original_dist)
            error = cv2.norm(valid_img[i], projected, cv2.NORM_L2) / len(projected)
            original_valid_error += error
            valid_count += 1
    original_valid_error /= valid_count if valid_count > 0 else 1

    # Step 4: 打印比较结果
    print("\n" + "="*40)
    print("标定结果比较")
    print("="*40)
    print(f"训练集重投影误差: {train_error:.4f} 像素")
    print(f"验证集重投影误差: {valid_error:.4f} 像素")
    print(f"原始参数在验证集上的误差: {original_valid_error:.4f} 像素")
    print(f"\n新参数矩阵:\n{new_mtx}")
    print(f"新畸变系数: {new_dist.flatten()}")

    # Step 5: 参数验证
    print("\n" + "="*40)
    print("原始参数验证")
    print("="*40)
    valid_original = validate_depth_camera(original_intrin)

    print("\n" + "="*40)
    print("新参数验证")
    print("="*40)
    valid_new = validate_depth_camera(new_intrin)

    # Step 6: 综合评估
    print("\n" + "="*40)
    print("最终评估结论")
    print("="*40)
    
    improvement = (original_valid_error - valid_error) / original_valid_error * 100
    print(f"验证集误差改进: {improvement:.1f}%")

    if valid_new:
        if valid_error < original_valid_error:
            print("✅ 新参数验证通过且验证集误差更低，建议采用新参数")
        else:
            print("⚠️ 新参数验证通过但验证集误差更高，需人工复核")
    else:
        if valid_error < original_valid_error:
            print("⚠️ 新参数未通过验证但验证集误差更低，需谨慎使用")
        else:
            print("❌ 新参数未通过验证且验证集误差更高，建议重新标定")

    # 保存新参数
    with open("new_depth_calibration.json", "w") as f:
        json.dump({
            "mtx": new_mtx.tolist(),
            "dist": new_dist.tolist(),
            "train_error": train_error,
            "valid_error": valid_error,
            "validation_passed": valid_new
        }, f, indent=2)

if __name__ == "__main__":
    main()
