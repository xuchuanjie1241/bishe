import cv2
import numpy as np


class CylinderUnwarper2:
    """
    用于将裁剪后的圆柱曲面图像展开为平面图像。
    基于透视投影原理，消除边缘的横向压缩和纵向拉伸变形。
    """

    def __init__(self):
        pass

    def unwarp(self, image, r, f):
        """
        执行圆柱展开变换。
        :param image: 输入图像 (H, W, C)，应为仅包含圆柱主体且垂直居中的裁剪图
        :param r: 圆柱体在图像中的像素半径 (通常接近图像宽度的一半)
        :param f: 校正后的等效焦距 (像素单位)
        :return: 展开后的平面图像
        """
        if image is None or image.size == 0:
            return image

        h, w = image.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # 1. 计算相机到圆柱中心的像素等效距离 D
        # 假设图像边缘恰好是视线与圆柱体相切的地方：tan(phi) = r / f
        # 根据勾股定理：D = sqrt(r^2 + f^2)
        D = np.sqrt(r ** 2 + f ** 2)

        # 2. 计算最大可见圆心角 theta_max
        # 相切处：cos(theta_max) = r / D
        theta_max = np.arccos(r / D)

        # 3. 确定展开后平面图像的尺寸
        # 宽度为可见圆弧的弧长：2 * r * theta_max
        w_flat = int(2 * r * theta_max)
        h_flat = h  # 保持中心处的高度比例不变

        # 4. 构造目标图像(展开图)的坐标网格
        u, v = np.meshgrid(np.arange(w_flat), np.arange(h_flat))

        # 将坐标系原点平移到展开图中心
        x_flat = u - w_flat / 2.0
        y_flat = v - h_flat / 2.0

        # 5. 计算展开图每个像素对应的圆柱面角度 theta
        theta = x_flat / r

        # 限制角度范围，避免因浮点误差产生无效投影
        theta = np.clip(theta, -theta_max, theta_max)

        # 6. 透视投影反变换 (将展开图坐标映射回原畸变图坐标)
        # 计算当前角度下，圆柱表面点到相机的深度 Z
        Z = D - r * np.cos(theta)

        # X 坐标映射 (处理横向压缩)
        map_x = f * (r * np.sin(theta)) / Z + cx

        # Y 坐标映射 (处理纵向透视变形，补偿由于深度 Z 增加造成的缩小效应)
        # 图像中心点(正对相机)的深度为 D - r
        map_y = y_flat * (D - r) / Z + cy

        # 7. 使用双线性插值进行重映射
        unwarped = cv2.remap(
            image,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return unwarped


import cv2
import numpy as np
import time


class CylinderUnwarper:
    def __init__(self):
        self.interpolation = cv2.INTER_LINEAR
        self.border_mode = cv2.BORDER_CONSTANT
        self.border_value = (0, 0, 0)

    def compute_mapping(self, h, w, r, f, center_y=None):
        """
        计算圆柱展开映射（可独立测试）
        :return: map_x, map_y, 及几何参数
        """
        cx = w / 2.0
        if center_y is not None:
            cy = center_y
        else:
            cy = h / 2.0
        D = np.sqrt(r ** 2 + f ** 2)
        theta_max = np.arccos(r / D)

        # 展开图尺寸
        w_flat = int(2 * r * theta_max)
        h_flat = h

        # 构造网格
        u, v = np.meshgrid(np.arange(w_flat), np.arange(h_flat))
        x_flat = u - w_flat / 2.0
        y_flat = v - cy

        # 逆映射计算
        theta = x_flat / r
        theta = np.clip(theta, -theta_max, theta_max)

        Z = D - r * np.cos(theta)
        map_x = f * (r * np.sin(theta)) / Z + cx
        map_y = y_flat * (D - r) / Z + cy

        geo_params = {
            'D': float(D),
            'theta_max': float(theta_max),
            'theta_range': [float(-theta_max), float(theta_max)],
            'w_flat': w_flat,
            'h_flat': h_flat,
            'arc_length_px': w_flat,
            'center_original': [float(cx), float(cy)],
            'focal_length_used': f,
            'radius_used': r
        }

        return map_x.astype(np.float32), map_y.astype(np.float32), geo_params

    def unwarp(self, image, r, f, center_y=None, return_details=False):
        """
        执行圆柱展开
        :return: 展开图像，或 (图像, details)
        """
        details = {
            'timing': {},
            'params': {
                'input_shape': list(image.shape),
                'radius': float(r),
                'focal_length': float(f)
            }
        }

        if image is None or image.size == 0:
            if return_details:
                return image, details
            return image

        t_start = time.time()
        h, w = image.shape[:2]

        # 计算映射
        t0 = time.time()
        map_x, map_y, geo_params = self.compute_mapping(h, w, r, f, center_y)
        details['timing']['mapping_compute_ms'] = (time.time() - t0) * 1000
        details['geometry'] = geo_params

        # 重映射
        t0 = time.time()
        unwarped = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=self.interpolation,
            borderMode=self.border_mode,
            borderValue=self.border_value
        )
        details['timing']['remap_ms'] = (time.time() - t0) * 1000

        # 创建质量图：标记边缘置信度（中心=1.0，边缘=0.0）
        h, w = unwarped.shape[:2]
        quality_map = np.ones((h, w), dtype=np.float32)

        # 形变分析：计算各区域的缩放因子
        t0 = time.time()
        # 中心区域缩放（应为1.0）
        center_x = geo_params['w_flat'] // 2
        sample_y = h // 2
        # 采样几个点计算局部Jacobian（近似）
        samples_x = [0, center_x // 2, center_x, center_x + center_x // 2, geo_params['w_flat'] - 1]
        local_scales = []
        for sx in samples_x:
            if 0 <= sx < geo_params['w_flat']:
                # 计算该位置对应的theta
                theta = (sx - center_x) / r
                # 几何缩放因子：d(original_x)/d(flat_x)
                Z = geo_params['D'] - r * np.cos(theta)
                scale = (f * r * np.cos(theta) / Z -
                         f * r * np.sin(theta) * (-r * np.sin(theta)) / (Z ** 2)) / r
                local_scales.append(abs(float(scale)))

        details['deformation_analysis'] = {
            'local_scales_at_samples': local_scales,
            'center_scale': local_scales[2] if len(local_scales) > 2 else 1.0,
            'edge_scale_ratio': (local_scales[0] / local_scales[2]
                                 if len(local_scales) > 2 and local_scales[2] != 0 else 1.0)
        }
        details['timing']['deformation_analysis_ms'] = (time.time() - t0) * 1000

        details['output_shape'] = list(unwarped.shape)
        details['timing']['total_ms'] = (time.time() - t_start) * 1000

        # 保存映射供可视化
        if return_details:
            details['intermediate'] = {
                'map_x': map_x,
                'map_y': map_y
            }

        if return_details:
            return unwarped, details
        return unwarped

    def validate_unwarping(self, checkerboard_img, r, f, grid_size=(10, 8)):
        """
        使用棋盘格验证展开精度（测试辅助方法）
        """
        unwarped, details = self.unwarp(checkerboard_img, r, f, return_details=True)

        gray = cv2.cvtColor(unwarped, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

        validation = {
            'checkerboard_found': ret,
            'corners_count': len(corners) if ret else 0
        }

        if ret:
            # 计算格距均匀性
            corners = corners.reshape(-1, 2)
            # 水平距离
            h_dists = []
            for i in range(grid_size[1]):
                for j in range(grid_size[0] - 1):
                    idx1 = i * grid_size[0] + j
                    idx2 = idx1 + 1
                    dist = np.linalg.norm(corners[idx1] - corners[idx2])
                    h_dists.append(dist)

            validation['mean_grid_spacing'] = float(np.mean(h_dists))
            validation['grid_spacing_cv'] = float(np.std(h_dists) / np.mean(h_dists))
            validation['is_uniform'] = validation['grid_spacing_cv'] < 0.05  # 5%变异系数阈值

        return unwarped, validation, details