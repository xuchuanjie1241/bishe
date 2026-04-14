import cv2
import numpy as np
from scipy import signal
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class MetricsCalculator:
    """评价指标计算器"""

    @staticmethod
    def reprojection_error(src_points, dst_points):
        """
        计算重投影误差
        :param src_points: 源点坐标 (N, 2)
        :param dst_points: 目标点坐标 (N, 2)
        :return: 平均误差, 标准差, 最大误差
        """
        errors = np.sqrt(np.sum((src_points - dst_points) ** 2, axis=1))
        return np.mean(errors), np.std(errors), np.max(errors)

    @staticmethod
    def line_straightness(points, fit_line=True):
        """
        计算点的直线度（拟合直线到点的平均距离）
        :param points: 点集 (N, 2)
        :return: 直线度误差（像素）
        """
        if len(points) < 2:
            return float('inf')

        # 使用PCA拟合直线
        mean = np.mean(points, axis=0)
        centered = points - mean
        _, _, vt = np.linalg.svd(centered)
        direction = vt[0]

        # 计算各点到直线的距离
        projections = centered @ direction
        projected_points = mean + np.outer(projections, direction)
        distances = np.sqrt(np.sum((points - projected_points) ** 2, axis=1))

        return np.mean(distances)

    @staticmethod
    def calculate_ssim(img1, img2):
        """计算SSIM"""
        # 转为灰度图
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        return ssim(img1, img2)

    @staticmethod
    def calculate_psnr(img1, img2):
        """计算PSNR"""
        return psnr(img1, img2, data_range=255)

    @staticmethod
    def seam_visibility_index(img, seam_x):
        """
        计算拼接缝可见度指数
        :param img: 融合后的图像
        :param seam_x: 接缝的x坐标位置
        :return: SVI值（越接近1越好）
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img.shape
        # 提取接缝附近区域
        window = 20
        left = max(0, seam_x - window)
        right = min(w, seam_x + window)

        seam_region = img[:, left:right]
        grad_x = cv2.Sobel(seam_region, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(seam_region, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 接缝处梯度与平均梯度的比值
        seam_grad = np.mean(gradient_magnitude[:, window - 2:window + 2])
        avg_grad = np.mean(gradient_magnitude)

        return seam_grad / (avg_grad + 1e-8)

    @staticmethod
    def cylindricity_preservation_rate(lines_angles):
        """
        计算圆柱度保持率
        :param lines_angles: 检测到的母线角度列表（度）
        :return: CPR值（%）
        """
        angles = np.array(lines_angles)
        std_angle = np.std(angles)
        max_allowed = 5.0  # 允许最大偏差5度
        cpr = (1 - std_angle / max_allowed) * 100
        return max(0, min(100, cpr))