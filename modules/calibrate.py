import numpy as np
import cv2
import glob

# --- 1. 参数设置 ---
# 棋盘格内角点数量 (行, 列) -> 注意：是交点数量，不是格子数
CHECKERBOARD = (7, 5)
# 棋盘格每个小方块的实际尺寸（单位可以是 mm 或 m，这决定了最后平移向量的单位）
SQUARE_SIZE = 15  # 假设每个方格 25mm

# 设置寻找亚像素角点的迭代准则
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 创建坐标容器
objpoints = [] # 在真实世界空间中的 3D 点
imgpoints = [] # 在图像平面中的 2D 点

# 准备 3D 世界坐标点，如 (0,0,0), (1,0,0), (2,0,0) ....,(6,4,0)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# --- 2. 加载图像并提取角点 ---
images = glob.glob('../calibration_images/*.jpg') # 图像路径

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        objpoints.append(objp)
        # 精细化角点坐标（亚像素级）
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 绘制并显示角点（可选）
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Corner Detection', img)
        cv2.imwrite('detection.jpg', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# --- 3. 运行标定 (核心步骤) ---
# ret: 重投影误差, mtx: 内参矩阵, dist: 畸变系数, rvecs: 旋转向量, tvecs: 平移向量
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n--- 标定结果 ---")
print(f"重投影误差 (RMS Error): {ret}")
print(f"相机内参矩阵 (Intrinsic Matrix):\n{mtx}")
print(f"畸变系数 (Distortion Coefficients):\n{dist}")

# --- 4. 畸变校正演示 ---
# 读取一张图片进行测试
test_img = cv2.imread(images[7])
h, w = test_img.shape[:2]
# 优化相机内参（可选，1表示保留所有像素，0表示只保留有效像素）
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# 方法 1: 使用 undistort 函数
dst = cv2.undistort(test_img, mtx, dist, None, new_camera_mtx)

# 裁剪（如果需要）
x, y, w_roi, h_roi = roi
dst = dst[y:y+h_roi, x:x+w_roi]
cv2.imwrite('calibrated_result.jpg', dst)
print("\n校正图片已保存为: calibrated_result.jpg")

# --- 5. 保存标定数据 ---
np.savez("camera_params.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)