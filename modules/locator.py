import cv2
import numpy as np


class CylinderLocator2:
    def __init__(self):
        self.ksize = 5
        self.ref_radius = None  # 基准半径
        self.target_radius = None  # 目标归一化半径

    def set_reference(self, radius_px):
        """设置基准半径（从第一张图获取）"""
        self.ref_radius = radius_px
        self.target_radius = radius_px  # 以第一张图为标准

    def process(self, img_raw, normalize=True):
        """
        处理图像，可选进行半径归一化
        normalize=True 时，会根据基准半径缩放图像
        """
        h_orig, w_orig = img_raw.shape[:2]

        # --- 原有边缘检测逻辑 ---
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.ksize, self.ksize), 0)

        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        lower = int(np.percentile(gradient_magnitude, 10))
        upper = int(np.percentile(gradient_magnitude, 50))
        lower, upper = max(10, lower), max(50, upper)

        edges = cv2.Canny(blurred, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # --- 检测半径 ---
        projection_x = np.sum(edges, axis=0)
        px_mean = np.mean(projection_x)
        px_std = np.std(projection_x)
        threshold = px_mean + 0.5 * px_std

        peaks_mask = projection_x > threshold
        peak_indices = np.where(peaks_mask)[0]

        if len(peak_indices) >= 2:
            from itertools import groupby
            groups = [(k, list(g)) for k, g in groupby(peak_indices,
                                                       key=lambda i, c=iter(peak_indices): i - next(c, i))]
            longest_group = max([g for _, g in groups], key=len, default=peak_indices)
            left_bound, right_bound = longest_group[0], longest_group[-1]
        else:
            margin = int(w_orig * 0.05)
            search_width = (w_orig - 2 * margin) // 3
            best_score, best_left = 0, margin
            for i in range(margin, w_orig - margin - search_width):
                score = np.sum(projection_x[i:i + search_width])
                if score > best_score:
                    best_score, best_left = score, i
            left_bound, right_bound = best_left, best_left + search_width

        radius_px = (right_bound - left_bound) / 2

        # --- 自适应垂直检测（保持原有逻辑）---
        roi_edges = edges[:, left_bound:right_bound]
        projection_y = np.sum(roi_edges, axis=1)

        y_margin = int(h_orig * 0.05)
        py_mean = np.mean(projection_y[y_margin:-y_margin])
        py_threshold = py_mean * 0.3

        y_indices = np.where(projection_y > py_threshold)[0]

        if len(y_indices) > 0:
            padding = int(h_orig * 0.02)
            top_bound = max(0, y_indices[0] - padding)
            bottom_bound = min(h_orig, y_indices[-1] + padding)
        else:
            top_bound, bottom_bound = h_orig // 4, 3 * h_orig // 4

        # --- 提取 ROI ---
        rect = img_raw[top_bound:bottom_bound, left_bound:right_bound]

        # --- 半径归一化---
        if normalize and self.target_radius is not None and radius_px > 0:
            scale_factor = self.target_radius / radius_px
            if abs(scale_factor - 1.0) > 0.01:  # 只在需要时缩放
                new_w = int(rect.shape[1] * scale_factor)
                new_h = int(rect.shape[0] * scale_factor)
                # 使用 INTER_AREA 缩小，INTER_CUBIC 放大，保持清晰度
                interp = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_CUBIC
                rect = cv2.resize(rect, (new_w, new_h), interpolation=interp)
                # 更新半径为归一化后的值
                radius_px = self.target_radius

        # 如果是第一张图（基准），自动设置参考
        if self.ref_radius is None:
            self.set_reference(radius_px)

        return rect, radius_px



import time
from collections import defaultdict


class CylinderLocator:
    def __init__(self):
        self.ksize = 5
        self.ref_radius = None
        self.target_radius = None
        # 可调参数（用于对比测试）
        self.edge_method = 'canny'  # 'canny', 'sobel'
        self.threshold_mode = 'adaptive'  # 'adaptive', 'otsu', 'fixed'

    def set_reference(self, radius_px):
        self.ref_radius = radius_px
        self.target_radius = radius_px

    def detect_edges(self, gray_img):
        """可测试的边缘检测方法"""
        blurred = cv2.GaussianBlur(gray_img, (self.ksize, self.ksize), 0)

        if self.edge_method == 'canny':
            # 自适应阈值
            if self.threshold_mode == 'adaptive':
                sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
                lower = int(np.percentile(gradient_magnitude, 10))
                upper = int(np.percentile(gradient_magnitude, 50))
                lower, upper = max(10, lower), max(50, upper)
            else:
                lower, upper = 50, 150

            edges = cv2.Canny(blurred, lower, upper)
            # 形态学闭运算连接断裂边缘
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        elif self.edge_method == 'sobel':
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            edges = np.uint8(np.absolute(sobelx))
            _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

        return edges, blurred

    def find_cylinder_bounds(self, edges, h, w):
        """定位圆柱体边界，返回详细统计"""
        stats = {}

        # 水平投影（找左右边界）
        projection_x = np.sum(edges, axis=0)
        stats['projection_x'] = projection_x.tolist()
        stats['projection_x_mean'] = float(np.mean(projection_x))
        stats['projection_x_std'] = float(np.std(projection_x))

        px_mean = np.mean(projection_x)
        px_std = np.std(projection_x)
        threshold = px_mean + 0.5 * px_std

        peaks_mask = projection_x > threshold
        peak_indices = np.where(peaks_mask)[0]

        if len(peak_indices) >= 2:
            from itertools import groupby
            groups = [(k, list(g)) for k, g in groupby(
                peak_indices, key=lambda i, c=iter(peak_indices): i - next(c, i)
            )]
            longest_group = max([g for _, g in groups], key=len, default=peak_indices)
            left_bound, right_bound = longest_group[0], longest_group[-1]
            detection_method = 'projection_grouping'
        else:
            # 回退策略
            margin = int(w * 0.05)
            search_width = (w - 2 * margin) // 3
            best_score, best_left = 0, margin
            for i in range(margin, w - margin - search_width):
                score = np.sum(projection_x[i:i + search_width])
                if score > best_score:
                    best_score, best_left = score, i
            left_bound, right_bound = best_left, best_left + search_width
            detection_method = 'sliding_window'

        radius_px = (right_bound - left_bound) / 2
        cx = (left_bound + right_bound) // 2

        # 垂直投影（找上下边界）
        roi_edges = edges[:, left_bound:right_bound]
        projection_y = np.sum(roi_edges, axis=1)
        stats['projection_y'] = projection_y.tolist()

        y_margin = int(h * 0.05)
        py_mean = np.mean(projection_y[y_margin:-y_margin])
        py_threshold = py_mean * 0.3

        y_indices = np.where(projection_y > py_threshold)[0]

        if len(y_indices) > 0:
            padding = int(h * 0.02)
            top_bound = max(0, y_indices[0] - padding)
            bottom_bound = min(h, y_indices[-1] + padding)
            detection_method_y = 'projection'
        else:
            top_bound, bottom_bound = h // 4, 3 * h // 4
            detection_method_y = 'default'

        bounds = {
            'left': int(left_bound),
            'right': int(right_bound),
            'top': int(top_bound),
            'bottom': int(bottom_bound),
            'center_x': int(cx),
            'center_y': int((top_bound + bottom_bound) // 2),
            'radius_px': float(radius_px),
            'width': int(right_bound - left_bound),
            'height': int(bottom_bound - top_bound)
        }

        stats.update({
            'detection_method_x': detection_method,
            'detection_method_y': detection_method_y,
            'threshold_used': float(threshold)
        })

        return bounds, stats

    def process(self, img_raw, normalize=True, return_details=False):
        """
        处理图像，可选返回详细统计
        :return: (rect_img, radius) 或 (rect_img, radius, details_dict)
        """
        details = {
            'timing': {},
            'params': {
                'edge_method': self.edge_method,
                'threshold_mode': self.threshold_mode,
                'normalize': normalize
            },
            'intermediate': {}
        }

        t_start = time.time()
        h_orig, w_orig = img_raw.shape[:2]
        details['input_size'] = [h_orig, w_orig]

        # 灰度转换
        t0 = time.time()
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        details['timing']['grayscale_ms'] = (time.time() - t0) * 1000

        # 边缘检测
        t0 = time.time()
        edges, blurred = self.detect_edges(gray)
        details['timing']['edge_detection_ms'] = (time.time() - t0) * 1000
        details['intermediate']['edges'] = edges
        details['intermediate']['blurred'] = blurred

        # 边界定位
        t0 = time.time()
        bounds, bound_stats = self.find_cylinder_bounds(edges, h_orig, w_orig)
        details['timing']['boundary_detection_ms'] = (time.time() - t0) * 1000
        details['bounds'] = bounds
        details['boundary_stats'] = bound_stats

        # 提取ROI
        rect = img_raw[bounds['top']:bounds['bottom'],
               bounds['left']:bounds['right']]
        radius_px = bounds['radius_px']

        left_bound = bounds['left']
        top_bound = bounds['top']

        center_in_rect = {
            'y': h_orig // 2 - top_bound,
            'radius': radius_px
        }

        # 归一化处理
        scale_info = {'applied': False, 'factor': 1.0}
        if normalize and self.target_radius is not None and radius_px > 0:
            scale_factor = self.target_radius / radius_px
            if abs(scale_factor - 1.0) > 0.01:
                t0 = time.time()
                new_w = int(rect.shape[1] * scale_factor)
                new_h = int(rect.shape[0] * scale_factor)
                interp = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_CUBIC
                rect = cv2.resize(rect, (new_w, new_h), interpolation=interp)
                radius_px = self.target_radius
                scale_info = {
                    'applied': True,
                    'factor': float(scale_factor),
                    'original_size': [int(rect.shape[0] / scale_factor), int(rect.shape[1] / scale_factor)],
                    'new_size': [new_h, new_w]
                }
                details['timing']['normalization_ms'] = (time.time() - t0) * 1000

        details['scale'] = scale_info

        # 自动设置参考（第一张图）
        if self.ref_radius is None:
            self.set_reference(radius_px)
            details['reference_set'] = True
        else:
            details['reference_set'] = False

        details['output_size'] = [rect.shape[0], rect.shape[1]]
        details['radius_px'] = float(radius_px)
        details['timing']['total_ms'] = (time.time() - t_start) * 1000

        if return_details:
            return rect, radius_px, center_in_rect, details
        return rect, radius_px, center_in_rect