import cv2
import numpy as np
import os

def get_chessboard_corners(image, pattern_size=(7,7)):
    """（保持不变）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not ret:
        return None
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    return corners

def reprojection_error(camera_matrix, distortion_coeffs, object_points, image_points):
    """（保持不变）"""
    _, rvecs, tvecs = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coeffs)
    projected_points, _ = cv2.projectPoints(object_points, rvecs, tvecs, camera_matrix, distortion_coeffs)
    errors = np.linalg.norm(projected_points - image_points, axis=2)
    return np.mean(errors)

def compare_intrinsics(ir_image_folder, 
                      camera_matrix1, distortion_coeffs1,
                      camera_matrix2, distortion_coeffs2,
                      camera_matrix3, distortion_coeffs3,
                      pattern_size=(7,7), square_size=15):
    """
    比较三个内参的重投影误差
    新增参数：
    :param camera_matrix3: 第三个内参矩阵
    :param distortion_coeffs3: 第三个畸变系数
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    ir_images = sorted([f for f in os.listdir(ir_image_folder) if f.endswith('.png')])

    # 初始化三个计数器
    count1, count2, count3 = 0, 0, 0

    for ir_file in ir_images:
        ir_image_path = os.path.join(ir_image_folder, ir_file)
        ir_image = cv2.imread(ir_image_path)

        corners = get_chessboard_corners(ir_image, pattern_size)
        if corners is None:
            print(f"未在图像 {ir_file} 中检测到棋盘格角点")
            continue

        # 计算三个内参的误差
        error1 = reprojection_error(camera_matrix1, distortion_coeffs1, objp, corners)
        error2 = reprojection_error(camera_matrix2, distortion_coeffs2, objp, corners)
        error3 = reprojection_error(camera_matrix3, distortion_coeffs3, objp, corners)

        # 找出最小误差的索引
        errors = [error1, error2, error3]
        min_index = np.argmin(errors)

        # 更新计数器
        if min_index == 0:
            count1 += 1
        elif min_index == 1:
            count2 += 1
        else:
            count3 += 1

        print(f"{ir_file}: 内参1误差={error1:.4f}, 内参2误差={error2:.4f}, 内参3误差={error3:.4f}")
        print(f"图像 {ir_file} 的内参{min_index+1}更好")

    # 输出最终结果
    total_images = len(ir_images)
    print(f"\n比较结果：")
    print(f"内参1胜率：{count1}/{total_images} ({count1/total_images*100:.2f}%)")
    print(f"内参2胜率：{count2}/{total_images} ({count2/total_images*100:.2f}%)")
    print(f"内参3胜率：{count3}/{total_images} ({count3/total_images*100:.2f}%)")

    max_count = max(count1, count2, count3)
    best_params = []
    if count1 == max_count: best_params.append("1")
    if count2 == max_count: best_params.append("2")
    if count3 == max_count: best_params.append("3")
    
    if len(best_params) > 1:
        print(f"综合结果：内参{'、'.join(best_params)} 并列最佳")
    else:
        print(f"综合结果：内参{best_params[0]} 最佳")

if __name__ == "__main__":
    # 定义三个内参
    camera_matrix1 = np.array([
        [385.9849853515625, 0, 323.7476806640625],
        [0, 385.9849853515625, 238.27999877929688],
        [0, 0, 1]], dtype=np.float32)
    distortion_coeffs1 = np.array([0, 0, 0, 0, 0], dtype=np.float32)

    camera_matrix2 = np.array([
        [390.64122761, 0, 450.30159532],
        [0, 392.57988967, 200.70517741],
        [0, 0, 1]], dtype=np.float32)
    distortion_coeffs2 = np.array([-0.09566405, 0.04583497, 0, 0, 0], dtype=np.float32)

    # 新增第三个内参
    camera_matrix3 = np.array([
        [337.21, 0, 320],
        [0, 432.97, 240],
        [0, 0, 1]], dtype=np.float32)
    distortion_coeffs3 = np.array([0, 0, 0, 0, 0], dtype=np.float32)

    ir_image_folder = "calibrition/ir_images"
    
    # 传入三个内参进行比较
    compare_intrinsics(ir_image_folder,
                      camera_matrix1, distortion_coeffs1,
                      camera_matrix2, distortion_coeffs2,
                      camera_matrix3, distortion_coeffs3)
