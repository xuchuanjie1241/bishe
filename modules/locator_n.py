import cv2
import numpy as np
class CylinderLocator3:
    def process(self, img_raw):
        # 1. 预处理
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        # 增强对比度，让透明瓶子边缘更明显
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 2. 提取边缘并寻找瓶身轮廓
        edges = cv2.Canny(gray, 50, 150)
        # 闭运算连接断开的边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 3. 寻找外接矩形或最小包围矩形
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img_raw, img_raw.shape[1] // 2

        # 假设最大的轮廓是瓶子
        c = max(contours, key=cv2.contourArea)

        # --- [新增] 摆正逻辑：获取最小外接矩形 (带旋转角度) ---
        rect_pts = cv2.minAreaRect(c)
        center, size, angle = rect_pts

        # 修正角度：minAreaRect的角度逻辑比较特殊
        if size[0] < size[1]:
            angle = angle
            w, h = size
        else:
            angle = angle + 90
            w, h = size[1], size[0]

        # 计算旋转矩阵并摆正
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_raw, M_rot, (img_raw.shape[1], img_raw.shape[0]), flags=cv2.INTER_CUBIC)

        # 4. 精确裁剪
        # 旋转后重新定位边界
        res_x = int(center[0] - w / 2)
        res_y = int(center[1] - h / 2)
        # 防止越界
        res_x, res_y = max(0, res_x), max(0, res_y)
        rectified = rotated[res_y:res_y + int(h), res_x:res_x + int(w)]

        # 5. [进阶] 透视矫正 (解决上大下小)
        # 如果展开效果依然差，可以在此处加入：
        # 根据侧边直线的斜率差异，用 cv2.warpPerspective 将瓶身拉为标准的平行矩形

        return rectified, w / 2

class CylinderLocator2:
    def process(self, img):
        """
        输入：原始图像
        输出：校正并裁剪后的圆柱体图像, 圆柱体在图中的像素半径
        """
        # 1. 预处理 (转灰度 + 高斯模糊)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # 3. 二值化 (替代 Canny)
        # 使用 OTSU 自动寻找阈值，假设背景是黑色的，瓶子总体偏亮
        # 如果你的背景是白的，瓶子是黑的，需要用 cv2.THRESH_BINARY_INV
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # --- 核心改进：形态学操作 ---
        # 定义一个较大的核（结构元素）。
        # 对于细长的圆柱体，可以用矩形核。
        # (21, 21) 的大小足够把一般的文字笔画“糊”在一起，变成实心块
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

        # 闭运算 = 先膨胀后腐蚀
        # 作用：连接断开的轮廓，填补内部的小黑洞（即文字部分）
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 也可以再加一步 腐蚀与膨胀的迭代，确保边缘平滑
        # morphed = cv2.erode(morphed, None, iterations=2)
        # morphed = cv2.dilate(morphed, None, iterations=2)

        # 4. 查找轮廓 (在形态学处理后的图上找)
        cnts, _ = cv2.findContours(morphed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnts:
            raise ValueError("未找到主体轮廓")

        # 3. 找到面积最大的轮廓 (假设圆柱体是主体)
        c = max(cnts, key=cv2.contourArea)

        # 4. 计算最小外接矩形 (获取中心点、宽高、旋转角度)
        rect = cv2.minAreaRect(c)
        (center_x, center_y), (w, h), angle = rect

        # 修正角度：minAreaRect的角度有时是-90~0，有时是0~90，需标准化为垂直
        if w > h:  # 如果检测到的是横着的矩形，说明角度偏差大，交换宽高
            angle = angle + 90
            w, h = h, w

        # 5. 位置校正 (旋转图像使圆柱垂直)
        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

        # 6. 裁剪 (Crop)
        # 既然已经旋转正了，直接根据宽高裁剪中心区域
        box = cv2.boxPoints(((center_x, center_y), (w, h), 0))  # 0度也就是正的
        box = np.int0(box)

        # 裁剪坐标计算
        start_x = max(0, int(center_x - w / 2))
        end_x = min(img.shape[1], int(center_x + w / 2))
        start_y = max(0, int(center_y - h / 2))
        end_y = min(img.shape[0], int(center_y + h / 2))

        cropped = rotated_img[start_y:end_y, start_x:end_x]

        # 返回裁剪图和半径 (宽度的一半)
        radius = w / 2
        return cropped, radius


import cv2
import numpy as np


class CylinderLocatorTilted:
    def __init__(self):
        self.ksize = 5
        self.hough_threshold = 80  # 霍夫直线检测阈值
        self.angle_tolerance = 20  # 倾斜角度容差（度）
        self.min_height_ratio = 0.3  # 圆柱体最小高度占图像比例

    def process(self, img_raw):
        h_orig, w_orig = img_raw.shape[:2]
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.ksize, self.ksize), 0)

        # 1. 边缘检测 + 形态学增强垂直线条
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 2. 霍夫直线检测（检测圆柱体的两条侧边）
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                                threshold=self.hough_threshold,
                                minLineLength=int(h_orig * 0.3),
                                maxLineGap=20)

        if lines is None or len(lines) < 2:
            # 回退到水平投影（垂直拍摄场景）
            return self._fallback_vertical(img_raw, edges, h_orig, w_orig)

        # 3. 筛选并分组：找左右两条近似平行的长直线（圆柱侧边）
        candidate_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算角度和长度
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 筛选接近垂直的直线（允许倾斜，但主要是纵向的）
            if 45 < angle < 90 and length > h_orig * 0.25:
                candidate_lines.append((x1, y1, x2, y2, angle, length))

        if len(candidate_lines) < 2:
            return self._fallback_vertical(img_raw, edges, h_orig, w_orig)

        # 4. 按 x 坐标排序，区分左右边界
        # 计算每条直线的中心 x 坐标
        candidate_lines.sort(key=lambda l: (l[0] + l[2]) / 2)

        # 取最左和最右的直线作为圆柱边界
        left_line = candidate_lines[0]
        right_line = candidate_lines[-1]

        # 检查平行度（两线角度差应在容差内）
        angle_diff = abs(left_line[4] - right_line[4])
        if angle_diff > self.angle_tolerance:
            # 角度差异过大，可能不是圆柱侧边，回退
            return self._fallback_vertical(img_raw, edges, h_orig, w_orig)

        # 5. 计算透视变换矩阵（将倾斜圆柱校正为正视图）
        # 提取两条直线的端点
        x1_l, y1_l, x2_l, y2_l = left_line[:4]
        x1_r, y1_r, x2_r, y2_r = right_line[:4]

        # 计算四个角点（透视校正的目标坐标）
        # 上边缘：取两线的上端点（y 较小者）
        top_y = min(y1_l, y2_l, y1_r, y2_r)
        # 下边缘：取两线的下端点（y 较大者）
        bottom_y = max(y1_l, y2_l, y1_r, y2_r)

        # 插值计算上下边界的左右 x 坐标
        def interp_x(x1, y1, x2, y2, target_y):
            if abs(y2 - y1) < 1e-6:
                return (x1 + x2) / 2
            return x1 + (x2 - x1) * (target_y - y1) / (y2 - y1)

        x_top_l = interp_x(x1_l, y1_l, x2_l, y2_l, top_y)
        x_bottom_l = interp_x(x1_l, y1_l, x2_l, y2_l, bottom_y)
        x_top_r = interp_x(x1_r, y1_r, x2_r, y2_r, top_y)
        x_bottom_r = interp_x(x1_r, y1_r, x2_r, y2_r, bottom_y)

        # 源四边形（倾斜的圆柱区域）
        pts_src = np.array([
            [x_top_l, top_y],
            [x_top_r, top_y],
            [x_bottom_r, bottom_y],
            [x_bottom_l, bottom_y]
        ], dtype=np.float32)

        # 目标四边形（校正为矩形）
        # 保持宽高比，宽度取平均
        width_px = int(((x_top_r - x_top_l) + (x_bottom_r - x_bottom_l)) / 2)
        height_px = int(bottom_y - top_y)

        pts_dst = np.array([
            [0, 0],
            [width_px, 0],
            [width_px, height_px],
            [0, height_px]
        ], dtype=np.float32)

        # 6. 透视变换（拉正圆柱体）
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        rect = cv2.warpPerspective(img_raw, M, (width_px, height_px))

        # 7. 半径计算（考虑透视校正后的像素-物理比例）
        # 这里假设已知相机参数或后续通过标定获取
        radius_px = width_px / 2

        # 计算倾斜角度（用于后续物理尺寸校正）
        tilt_angle = np.mean([left_line[4], right_line[4]])

        return rect, radius_px, {
            'tilt_angle': tilt_angle,
            'transform_matrix': M,
            'src_points': pts_src,
            'dst_points': pts_dst,
            'correction_applied': True
        }

    def _fallback_vertical(self, img_raw, edges, h_orig, w_orig):
        """垂直拍摄的回退方案"""
        projection_x = np.sum(edges, axis=0)
        threshold = np.mean(projection_x) * 2
        peaks = np.where(projection_x > threshold)[0]

        if len(peaks) >= 2:
            left_bound, right_bound = peaks[0], peaks[-1]
        else:
            margin = int(w_orig * 0.05)
            left_bound, right_bound = margin, w_orig - margin

        projection_y = np.sum(edges[:, left_bound:right_bound], axis=1)
        y_indices = np.where(projection_y > np.mean(projection_y))[0]

        if len(y_indices) > 0:
            top_bound, bottom_bound = max(0, y_indices[0] - 10), min(h_orig, y_indices[-1] + 10)
        else:
            top_bound, bottom_bound = int(h_orig * 0.2), int(h_orig * 0.8)

        rect = img_raw[top_bound:bottom_bound, left_bound:right_bound]
        radius_px = (right_bound - left_bound) / 2

        return rect, radius_px, {
            'tilt_angle': 0,
            'correction_applied': False,
            'bounds': (left_bound, top_bound, right_bound, bottom_bound)
        }


class CylinderLocatorTiltedV2:
    """
    更鲁棒的版本：支持部分遮挡、不完全可见的圆柱体
    使用轮廓拟合而非强制要求两条完整边线
    """

    def __init__(self):
        self.ksize = 5
        self.min_contour_area = 500

    def process(self, img_raw):
        h_orig, w_orig = img_raw.shape[:2]
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.ksize, self.ksize), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0, {'error': 'No contours found'}

        # 筛选可能为圆柱侧面的轮廓（长条形、纵向）
        candidates = []
        for cnt in contours:
            if len(cnt) < 5:
                continue
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            # 最小外接矩形
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect

            # 长宽比（圆柱体应该是高 > 宽）
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            if aspect_ratio > 2 and max(w, h) > h_orig * 0.3:
                candidates.append({
                    'rect': rect,
                    'center': (cx, cy),
                    'size': (w, h),
                    'angle': angle,
                    'contour': cnt,
                    'area': area
                })

        if not candidates:
            return None, 0, {'error': 'No valid cylinder contours'}

        # 合并相近的轮廓（圆柱的左右边可能是分开的轮廓）
        candidates.sort(key=lambda x: x['center'][0])

        # 如果只有一个长轮廓，直接用它
        if len(candidates) == 1:
            best = candidates[0]
        else:
            # 找两个中心 x 距离最大且面积较大的（左右边界）
            best_pair = None
            max_dist = 0
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    dist = abs(candidates[i]['center'][0] - candidates[j]['center'][0])
                    if dist > max_dist and dist > w_orig * 0.1:
                        max_dist = dist
                        best_pair = (candidates[i], candidates[j])

            if best_pair:
                # 合并两个轮廓的边界框
                left, right = sorted(best_pair, key=lambda x: x['center'][0])
                # 创建包含两者的 ROI
                x1 = int(left['center'][0] - left['size'][0] / 2)
                x2 = int(right['center'][0] + right['size'][0] / 2)
                y1 = int(min(left['center'][1] - left['size'][1] / 2,
                             right['center'][1] - right['size'][1] / 2))
                y2 = int(max(left['center'][1] + left['size'][1] / 2,
                             right['center'][1] + right['size'][1] / 2))

                # 提取 ROI
                x1, x2 = max(0, x1), min(w_orig, x2)
                y1, y2 = max(0, y1), min(h_orig, y2)
                rect_img = img_raw[y1:y2, x1:x2]
                radius_px = (x2 - x1) / 2

                return rect_img, radius_px, {
                    'tilt_angle': (left['angle'] + right['angle']) / 2,
                    'bounds': (x1, y1, x2, y2),
                    'method': 'dual_contour'
                }
            else:
                best = candidates[0]

        # 单轮廓处理
        (cx, cy), (w, h), angle = best['rect']
        box = cv2.boxPoints(best['rect'])
        box = np.int0(box)

        # 提取旋转矩形区域
        width, height = int(w), int(h)
        if width < height:
            width, height = height, width
            angle += 90

        src_pts = box.astype("float32")
        # 重新排序角点：左上、右上、右下、左下
        src_pts = self._order_points(src_pts)

        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        rect = cv2.warpPerspective(img_raw, M, (width, height))

        radius_px = width / 2

        return rect, radius_px, {
            'tilt_angle': angle,
            'transform_matrix': M,
            'method': 'single_contour',
            'box': box
        }

    def _order_points(self, pts):
        """按顺序排列角点：左上、右上、右下、左下"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下
        rect[1] = pts[np.argmin(diff)]  # 右上
        rect[3] = pts[np.argmax(diff)]  # 左下

        return rect



class CylinderLocator6:
    def __init__(self):
        self.ksize = 5  # 滤波核

    def process(self, img_raw):
        h_orig, w_orig = img_raw.shape[:2]
        # 1. 预处理：转灰度并高斯模糊，消除手机拍摄的高频噪点
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.ksize, self.ksize), 0)

        v_median = np.median(gray)
        lower = int(max(0, (1.0 - 0.33) * v_median))
        upper = int(min(255, (1.0 + 0.33) * v_median))
        edges = cv2.Canny(blurred, lower, upper)
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        projection_x = np.sum(edges, axis=0)

        threshold = np.mean(projection_x) * 3
        peaks = np.where(projection_x > threshold)[0]

        if len(peaks) >= 2:
            left_bound = peaks[0]
            right_bound = peaks[-1]
        else:
            margin = int(len(projection_x) * 0.05)
            mid = len(projection_x) // 2
            left_bound = np.argmax(projection_x[margin:mid]) + margin
            right_bound = np.argmax(projection_x[mid:-margin]) + mid

        # --- 步骤 3: 确定上下边界 (高度裁剪) ---
        # 统计横向边缘投影
        projection_y = np.sum(edges[:, left_bound:right_bound], axis=1)
        # 忽略顶部和底部 10% 的干扰区域，寻找瓶身有效高度
        y_indices = np.where(projection_y > np.mean(projection_y))[0]
        top_bound, bottom_bound = y_indices[0], y_indices[-1]

        # 提取最终 ROI
        rect = img_raw[top_bound:bottom_bound, left_bound:right_bound]
        radius_px = (right_bound - left_bound) / 2

        return rect, radius_px


class CylinderLocator4:
    def __init__(self):
        self.ksize = 5
        self.min_radius_ratio = 0.15  # 最小半径占图像宽度比例
        self.edge_threshold_ratio = 0.5  # 边缘投影阈值系数

    def process(self, img_raw):
        h_orig, w_orig = img_raw.shape[:2]

        # 1. 预处理：灰度 + 高斯模糊
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.ksize, self.ksize), 0)

        # 2. 改进的 Canny：使用自适应阈值或 Otsu 辅助
        # 方案 A：基于梯度的自适应阈值
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # 使用梯度分布的百分位数确定 Canny 阈值
        lower = int(np.percentile(gradient_magnitude, 10))
        upper = int(np.percentile(gradient_magnitude, 50))
        lower, upper = max(10, lower), max(50, upper)  # 保护下限

        edges = cv2.Canny(blurred, lower, upper)

        # 3. 形态学闭运算连接断裂的垂直边缘（圆柱体侧边）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 4. 水平投影找左右边界（圆柱宽度）
        projection_x = np.sum(edges, axis=0)

        # 自适应阈值：基于投影的统计特性
        px_mean = np.mean(projection_x)
        px_std = np.std(projection_x)
        threshold = px_mean + self.edge_threshold_ratio * px_std

        # 找到超过阈值的连续区域
        peaks_mask = projection_x > threshold
        peak_indices = np.where(peaks_mask)[0]

        if len(peak_indices) >= 2:
            # 找最宽的有效连续区域
            from itertools import groupby
            groups = [(k, list(g)) for k, g in groupby(peak_indices,
                                                       key=lambda i, c=iter(peak_indices): i - next(c, i))]

            # 选择最长的连续段作为圆柱体区域
            longest_group = max([g for _, g in groups], key=len, default=peak_indices)
            left_bound, right_bound = longest_group[0], longest_group[-1]

            # 宽度验证
            width = right_bound - left_bound
            if width < w_orig * self.min_radius_ratio * 2:
                raise ValueError(f"检测到的宽度({width}px)过小，可能不是完整圆柱体")
        else:
            # 回退：使用边缘密度最高的区域
            margin = int(w_orig * 0.05)
            search_width = (w_orig - 2 * margin) // 3

            # 滑动窗口找最大边缘密度
            best_score, best_left = 0, margin
            for i in range(margin, w_orig - margin - search_width):
                score = np.sum(projection_x[i:i + search_width])
                if score > best_score:
                    best_score = score
                    best_left = i

            left_bound = best_left
            right_bound = best_left + search_width

        # 5. 垂直投影找上下边界（在已确定的宽度范围内）
        roi_edges = edges[:, left_bound:right_bound]
        projection_y = np.sum(roi_edges, axis=1)

        # 使用边缘保护 + 自适应阈值
        y_margin = int(h_orig * 0.05)
        py_mean = np.mean(projection_y[y_margin:-y_margin])
        py_threshold = py_mean * 0.3  # 降低阈值，避免漏检

        y_indices = np.where(projection_y > py_threshold)[0]

        if len(y_indices) > 0:
            # 扩展边界，确保包含完整瓶身
            padding = int(h_orig * 0.02)
            top_bound = max(0, y_indices[0] - padding)
            bottom_bound = min(h_orig, y_indices[-1] + padding)
        else:
            # 完全失败时使用图像中心区域
            top_bound = h_orig // 4
            bottom_bound = 3 * h_orig // 4

        # 6. 提取 ROI 并计算半径
        rect = img_raw[top_bound:bottom_bound, left_bound:right_bound]
        radius_px = (right_bound - left_bound) / 2


        return rect, radius_px


class CylinderLocator2:
    def __init__(self):
        self.f = 1000  # 预估焦距 (像素级)，实际应用中需标定
        self.D_ratio = 5.0  # D/R 的比例，影响拉伸程度

    def get_rotation_angle(self, edges):
        """通过检测最长的直线来计算偏转角度"""
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=10)
        if lines is None:
            return 0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # 我们寻找接近垂直的线（90度或-90度）
            if abs(angle) > 70:
                # 转换为偏离垂直线的角度
                correction = angle - 90 if angle > 0 else angle + 90
                angles.append(correction)

        return np.median(angles) if angles else 0

    def process(self, img_raw):
        h, w = img_raw.shape[:2]

        # 1. 旋转矫正
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        angle = self.get_rotation_angle(edges)

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_rotated = cv2.warpAffine(img_raw, M, (w, h), flags=cv2.INTER_CUBIC)
        gray_rotated = cv2.warpAffine(gray, M, (w, h))

        # 2. 左右边界定位 (投影法)
        # 重新计算旋转后的边缘以获得更准的投影
        edges_rot = cv2.Canny(gray_rotated, 50, 150)
        x_projection = np.sum(edges_rot, axis=0)

        # 寻找左右最外侧的强边缘
        threshold = np.max(x_projection) * 0.3
        peaks = np.where(x_projection > threshold)[0]
        if len(peaks) < 2:
            return None, 0

        left, right = peaks[0], peaks[-1]

        # 3. 上下边界定位 (基于梯度变化)
        # 圆柱体顶部和底部通常有弧形边缘或剧烈亮度变化
        y_projection = np.sum(edges_rot[:, left:right], axis=1)
        y_threshold = np.mean(y_projection) * 0.8
        y_peaks = np.where(y_projection > y_threshold)[0]

        top, bottom = y_peaks[0], y_peaks[-1]

        # 4. 裁剪 ROI
        roi = img_rotated[top:bottom, left:right]
        radius_px = (right - left) / 2

        return roi, radius_px