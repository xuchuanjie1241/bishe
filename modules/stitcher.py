# stitcher.py (修改版)
from enum import Enum

import cv2
import numpy as np
import time

class TransformType(Enum):
    """变换类型枚举"""
    HOMOGRAPHY = "homography"      # 透视变换 (8自由度)
    AFFINE = "affine"              # 仿射变换 (6自由度)
    TRANSLATION = "translation"    # 纯平移 (2自由度) - 推荐用于圆柱面

class SimpleStitcher:
    def __init__(self, method='laplacian',
                 match_strategy='full',
                 transform_type=TransformType.AFFINE,
                 edge_discard_ratio=0,
                 y_alignment_strength=1.0):
        """
        :param method: 融合方法 ('average', 'laplacian', 'poisson')
        :param match_strategy: 匹配策略 ('lowe', 'vertical', 'ransac', 'full')
                              'lowe': 仅Lowe比率测试
                              'vertical': Lowe + 垂直约束
                              'ransac': Lowe + RANSAC (无垂直约束)
                              'full': Lowe + 垂直约束 + RANSAC (默认)
        """
        self.method = method
        self.match_strategy = match_strategy
        self.transform_type = transform_type
        self.edge_discard = edge_discard_ratio
        self.y_alignment_strength = y_alignment_strength
        self.y_tolerance_factor = 0.03  # Y方向容差因子 (高度一致性约束)
        self.min_translation_x = 0.1    # 最小水平位移 (避免完全重叠)
        self.max_vertical_shift = 0.1  # 最大允许垂直偏移 (相对于图像高度)
        self.cumulative_ty = 0.0

    def detect_and_compute(self, img):
        """特征提取，支持不同OpenCV版本"""
        try:
            sift = cv2.SIFT_create()
        except AttributeError:
            sift = cv2.xfeatures2d.SIFT_create()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        kp, des = sift.detectAndCompute(gray, None)
        return kp, des, gray

    def preprocess_for_stitching(self, img):
        """预处理：创建有效区域掩码，边缘降权"""
        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)

        # 左右边缘渐变降权（羽化）
        fade_width = int(w * self.edge_discard)
        fade_height = int(h * 0.02)
        if fade_width > 0:
            # 左边缘线性渐变 0->1
            mask[:, :fade_width] = np.linspace(0, 1, fade_width)
            # 右边缘线性渐变 1->0
            mask[:, -fade_width:] = np.linspace(1, 0, fade_width)
            '''
            x = np.linspace(0, np.pi / 2, fade_width)
            mask[:, :fade_width] = 1 - np.sin(x) * 0.9  # 保留一点信息，不完全归零

            # 右边缘
            mask[:, -fade_width:] = 1 - np.sin(x[::-1]) * 0.9
            '''

        '''
        if fade_height > 0:
            # 上下边缘轻微降权 (处理圆柱面展开的边缘弧度)
            y = np.linspace(0, np.pi / 2, fade_height)
            mask[:fade_height, :] *= (1 - np.sin(y))[:, None] * 0.5 + 0.5
            mask[-fade_height:, :] *= (1 - np.sin(y[::-1]))[:, None] * 0.5 + 0.5
        '''
        return mask

    def match_features(self, des1, des2, kp1, kp2, img_height, strategy=None):
        """
        特征匹配与筛选（可独立测试）
        :return: 筛选后的匹配点列表，以及统计信息
        """
        if strategy is None:
            strategy = self.match_strategy

        # KNN匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        stats = {
            'initial_matches': len(matches),
            'strategy': strategy,
            'lowe_rejected': 0,
            'vertical_rejected': 0,
            'ransac_rejected': 0
        }

        # 阶段1: Lowe比率测试
        lowe_matches = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                lowe_matches.append(m)
            else:
                stats['lowe_rejected'] += 1

        stats['after_lowe'] = len(lowe_matches)

        if strategy == 'lowe':
            return lowe_matches, stats

        # 阶段2: 垂直几何约束
        vertical_matches = []
        y_tolerance = img_height * self.y_tolerance_factor

        for m in lowe_matches:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt

            if abs(pt1[1] - pt2[1]) < y_tolerance:
                vertical_matches.append(m)
            else:
                stats['vertical_rejected'] += 1

        stats['after_vertical'] = len(vertical_matches)

        if strategy == 'vertical':
            return vertical_matches, stats

        # 阶段3: RANSAC几何验证
        if strategy in ['ransac', 'full']:
            # 根据策略选择输入
            input_matches = vertical_matches if strategy == 'full' else lowe_matches

            if len(input_matches) < 4:
                return input_matches, stats

            src_pts = np.float32([kp2[m.trainIdx].pt for m in input_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in input_matches]).reshape(-1, 1, 2)

            # 使用Homography进行RANSAC
            # 根据变换类型选择RANSAC方法
            if self.transform_type == TransformType.TRANSLATION:
                # 纯平移模型：使用自定义RANSAC
                M, mask = self._estimate_translation_ransac(src_pts, dst_pts)
            elif self.transform_type == TransformType.AFFINE:
                M, mask = cv2.estimateAffine2D(src_pts, dst_pts,
                                               method=cv2.RANSAC,
                                               ransacReprojThreshold=3.0)
            else:  # HOMOGRAPHY
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if mask is not None:
                ransac_matches = []
                for i, m in enumerate(input_matches):
                    if mask[i][0]:
                        ransac_matches.append(m)
                    else:
                        stats['ransac_rejected'] += 1
                stats['after_ransac'] = len(ransac_matches)
                return ransac_matches, stats

        return vertical_matches, stats

    def _estimate_translation_ransac(self, src_pts, dst_pts, max_iters=400, threshold=10.0):
        """
        专用平移变换RANSAC估计（圆柱面优化）
        假设：只有水平平移 tx，垂直平移 ty 很小
        """

        if len(src_pts) < 2:
            return None, None

        src_pts = src_pts.reshape(-1, 2)
        dst_pts = dst_pts.reshape(-1, 2)

        best_inliers = []
        best_tx, best_ty = 0, 0

        ty_candidates = dst_pts[:, 1] - src_pts[:, 1]
        ty_median = np.median(ty_candidates)

        for _ in range(max_iters):
            # 随机选择一对点计算平移
            idx = np.random.randint(0, len(src_pts))
            tx = dst_pts[idx, 0] - src_pts[idx, 0]
            ty = dst_pts[idx, 1] - src_pts[idx, 1]

            # 圆柱面约束：垂直位移应该很小
            if abs(ty) > dst_pts.shape[0] * self.max_vertical_shift:
                continue
            '''
            if abs(ty - ty_median) > src_pts.shape[0] * self.max_vertical_shift:
                continue
            '''
            # 计算所有点的重投影误差
            projected = src_pts + np.array([tx, ty])
            errors = np.linalg.norm(projected - dst_pts, axis=1)

            inliers = np.where(errors < threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_tx, best_ty = tx, ty

        if len(best_inliers) < 4:
            return None, None

        # 用所有内点重新估计平移
        final_tx = np.mean(dst_pts[best_inliers, 0] - src_pts[best_inliers, 0])
        final_ty = np.mean(dst_pts[best_inliers, 1] - src_pts[best_inliers, 1])

        final_ty *= self.y_alignment_strength
        print(f"Final translation: ({final_tx}, {final_ty})")

        # 构建3x3平移矩阵
        M = np.array([[1, 0, final_tx],
                      [0, 1, final_ty],
                      [0, 0, 1]], dtype=np.float32)

        # 创建mask
        mask = np.zeros((len(src_pts), 1), dtype=np.uint8)
        mask[best_inliers] = 1

        return M, mask

    def estimate_transform(self, kp1, kp2, matches, direction=1):
        """
        估计几何变换矩阵
        :return: 变换矩阵M, 统计信息
        """
        if len(matches) < 4:
            return None, {'error': ' insufficient matches'}

        if direction == 1:
            src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            base_img_idx, warp_img_idx = 0, 1  # 用于记录
        else:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            base_img_idx, warp_img_idx = 1, 0

        # 使用Homography
        # 根据变换类型选择估计方法
        if self.transform_type == TransformType.TRANSLATION:
            M, mask = self._estimate_translation_ransac(src_pts, dst_pts)
            matrix_type = 'Translation'
        elif self.transform_type == TransformType.AFFINE:
            M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC,
                                           ransacReprojThreshold=3.0)
            matrix_type = 'Affine'
        else:  # HOMOGRAPHY
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matrix_type = 'Homography'

        stats = {
            'matrix_type': matrix_type,
            'inliers': int(np.sum(mask)) if mask is not None else 0,
            'outliers': len(matches) - int(np.sum(mask)) if mask is not None else 0,
            'direction': direction,
            'transform_params': self._extract_transform_params(M) if M is not None else None
        }

        return M, stats

    def _extract_transform_params(self, M):
        """提取变换参数用于调试"""
        if M is None:
            return None
        if self.transform_type == TransformType.TRANSLATION:
            return {'tx': M[0, 2], 'ty': M[1, 2]}
        elif self.transform_type == TransformType.AFFINE:
            return {
                'tx': M[0, 2], 'ty': M[1, 2],
                'scale_x': np.sqrt(M[0,0]**2 + M[0,1]**2),
                'rotation': np.arctan2(M[1,0], M[0,0]) * 180 / np.pi
            }
        else:
            return {'matrix': M.tolist()}

    def _transform_corners(self, corners, M):
        """
        使用合适的变换方法变换角点
        :param corners: 角点数组 (N, 1, 2)
        :param M: 变换矩阵 (2x3 for Affine/Translation, 3x3 for Homography)
        :return: 变换后的角点
        """
        if self.transform_type in [TransformType.AFFINE, TransformType.TRANSLATION]:
            # 仿射/平移变换使用 cv2.transform (接受2x3矩阵)
            return cv2.transform(corners, M)
        else:
            # 透视变换使用 cv2.perspectiveTransform (需要3x3矩阵)
            return cv2.perspectiveTransform(corners, M)

    def stitch(self, img_left, img_right, direction=1, return_details=False, base_offset_y=0):
        """
        执行拼接
        :param return_details: 如果为True，返回字典包含中间结果
        :return: 拼接结果图像，或 (图像, 详细信息字典)
        """
        details = {
            'timing': {},
            'matching': {},
            'transform': {},
            'fusion': {}
        }

        t_start = time.time()

        # 1. 特征提取
        t0 = time.time()
        kp1, des1, gray1 = self.detect_and_compute(img_left)
        kp2, des2, gray2 = self.detect_and_compute(img_right)
        details['timing']['feature_extraction_ms'] = (time.time() - t0) * 1000
        details['matching']['keypoints_left'] = len(kp1)
        details['matching']['keypoints_right'] = len(kp2)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            if return_details:
                return None, details
            return None

        # 2. 特征匹配
        t0 = time.time()
        good_matches, match_stats = self.match_features(
            des1, des2, kp1, kp2, img_left.shape[0], self.match_strategy
        )
        details['timing']['matching_ms'] = (time.time() - t0) * 1000
        details['matching'].update(match_stats)
        details['matching']['final_matches'] = len(good_matches)

        if len(good_matches) < 4:
            print(f"匹配点不足: {len(good_matches)}")
            if return_details:
                return None, details
            return None

        # 3. 几何变换估计
        t0 = time.time()
        M, transform_stats = self.estimate_transform(kp1, kp2, good_matches, direction)
        print(f"变换矩阵: {M}")
        details['timing']['transform_estimation_ms'] = (time.time() - t0) * 1000
        details['transform'].update(transform_stats)

        if M is None:
            if return_details:
                return None, details
            return None

        # 4. 计算画布与变换
        tx, ty = M[0, 2], M[1, 2]
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]

        if direction == 1:
            base_img, warp_img = img_left, img_right
            base_h, base_w = h1, w1
            warp_h, warp_w = h2, w2
        else:
            base_img, warp_img = img_right, img_left
            base_h, base_w = h2, w2
            warp_h, warp_w = h1, w1

        # 提取变换参数 (tx, ty)
        if self.transform_type in [TransformType.AFFINE, TransformType.TRANSLATION]:
            # 2x3矩阵: [[a, b, tx], [c, d, ty]]
            tx, ty = M[0, 2], M[1, 2]
        else:
            # 3x3矩阵，需要从齐次坐标转换
            tx, ty = M[0, 2], M[1, 2]

        # 对于平移和仿射变换，计算更简单
        if self.transform_type in [TransformType.TRANSLATION, TransformType.AFFINE]:
            if direction == 1:  # 右图变换到左图坐标系
                # 计算包围盒
                corners_warp = np.float32([[0, 0], [0, warp_h], [warp_w, warp_h], [warp_w, 0]]).reshape(-1, 1, 2)
                transformed_corners = self._transform_corners(corners_warp, M)

                # 左图位置: [0, base_w] x [0, base_h]
                # 右图变换后位置: transformed_corners
                all_points = np.concatenate((
                    np.float32([[0, 0], [0, base_h], [base_w, base_h], [base_w, 0]]).reshape(-1, 1, 2),
                    transformed_corners
                ), axis=0)

                [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
                [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

                output_w = max(x_max - x_min, base_w)
                output_h = max(y_max - y_min, base_h)

                # 基准图(左图)在画布中的位置
                roi_x = int(0 - x_min)
                roi_y = int(0 - y_min)

                # 调整变换矩阵，使图像平移到正确位置
                M_adjusted = M.copy()
                M_adjusted[0, 2] = tx - x_min
                M_adjusted[1, 2] = ty - y_min

            else:  # 左图变换到右图坐标系 (direction=-1)
                # 计算包围盒
                corners_warp = np.float32([[0, 0], [0, warp_h], [warp_w, warp_h], [warp_w, 0]]).reshape(-1, 1, 2)
                transformed_corners = self._transform_corners(corners_warp, M)

                # 右图位置: [0, base_w] x [0, base_h]
                # 左图变换后位置: transformed_corners
                all_points = np.concatenate((
                    np.float32([[0, 0], [0, base_h], [base_w, base_h], [base_w, 0]]).reshape(-1, 1, 2),
                    transformed_corners
                ), axis=0)

                [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
                [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

                output_w = max(x_max - x_min, base_w)
                output_h = max(y_max - y_min, base_h)

                # 基准图(右图)在画布中的位置
                roi_x = int(0 - x_min)
                roi_y = int(0 - y_min)

                # 调整变换矩阵
                M_adjusted = M.copy()
                M_adjusted[0, 2] = tx - x_min
                M_adjusted[1, 2] = ty - y_min

            M = M_adjusted

        else:  # HOMOGRAPHY
            # 透视变换的画布计算
            corners_base = np.float32([[0, 0], [0, base_h], [base_w, base_h], [base_w, 0]]).reshape(-1, 1, 2)
            corners_warp = np.float32([[0, 0], [0, warp_h], [warp_w, warp_h], [warp_w, 0]]).reshape(-1, 1, 2)

            transformed_corners = cv2.perspectiveTransform(corners_warp, M)

            all_points = np.concatenate((corners_base, transformed_corners), axis=0)
            [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

            translation_dist = [-x_min, -y_min]
            H_translation = np.array([[1, 0, translation_dist[0]],
                                      [0, 1, translation_dist[1]],
                                      [0, 0, 1]])
            M = H_translation.dot(M)
            roi_x, roi_y = translation_dist[0], translation_dist[1]
            output_w = max(x_max - x_min, roi_x + base_w)
            output_h = max(y_max - y_min, roi_y + base_h)

        print(f"画布: {output_w}x{output_h}, 基准图位置: ({roi_x}, {roi_y}), 基准图尺寸: {base_w}x{base_h}")

        '''
        # 对于平移变换，计算更简单
        if self.transform_type == TransformType.TRANSLATION:
            tx, ty = M[0, 2], M[1, 2]

            # 确定画布大小
            if direction == 1:  # right image warped to left
                output_w = int(max(w1, w2 + tx))
                output_h = int(max(h1, h2 + abs(ty)))
                roi_x, roi_y = 0, max(0, int(ty))
            else:
                output_w = int(max(w1 + abs(tx), w2))
                output_h = int(max(h1 + abs(ty), h2))
                roi_x, roi_y = max(0, int(tx)), max(0, int(ty))

            # 确保画布足够大
            output_w = max(output_w, w1 + w2)
            output_h = max(output_h, h1 + h2 // 2 + ty)
            print(f"roi_x: {roi_x}, roi_y: {roi_y}")
            
                        # 考虑基准偏移后的角点
            left_top = 0 + base_offset_y
            left_bottom = h1 + base_offset_y
            right_top = ty  # 右图相对左图的偏移
            right_bottom = h2 + ty
            
            y_min = min(left_top, right_top)
            y_max = max(left_bottom, right_bottom)
            x_min = min(0, tx)
            x_max = max(w1, w2 + tx)
            
            output_w = int(x_max - x_min)
            output_h = int(y_max - y_min)
            
            # 左图位置（加上基准偏移）
            roi_x = int(0 - x_min)
            roi_y = int(base_offset_y - y_min)
            
            # 右图变换：应用ty实现Y对齐
            affine_M = np.array([
                [1, 0, tx - x_min],
                [1, 0, ty - y_min]  # 关键：这里包含ty，实现Y方向对齐
            ], dtype=np.float32)
            
            # 记录本次的Y偏移供下次使用
            current_ty_offset = ty
            
        else:
            # 原有透视/仿射变换的画布计算
            corners_base = np.float32([[0, 0], [0, base_h], [base_w, base_h], [base_w, 0]]).reshape(-1, 1, 2)
            corners_warp = np.float32([[0, 0], [0, warp_h], [warp_w, warp_h], [warp_w, 0]]).reshape(-1, 1, 2)

            if self.transform_type == TransformType.TRANSLATION:
                transformed_corners = corners_warp + np.array([M[0, 2], M[1, 2]])
            else:
                transformed_corners = cv2.perspectiveTransform(corners_warp, M)

            all_points = np.concatenate((corners_base, transformed_corners), axis=0)
            [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

            translation_dist = [-x_min, -y_min]
            H_translation = np.array([[1, 0, translation_dist[0]],
                                      [0, 1, translation_dist[1]],
                                      [0, 0, 1]])
            M = H_translation.dot(M)
            roi_x, roi_y = translation_dist[0], translation_dist[1]
            output_w = max(x_max - x_min, roi_x + base_w)
            output_h = max(y_max - y_min, roi_y + base_h)
        '''

        # 5. 图像变换
        t0 = time.time()

        # 使用平移变换时，使用更快的warpAffine
        if self.transform_type in [TransformType.AFFINE, TransformType.TRANSLATION]:
            # 平移矩阵提取
            warped_img = cv2.warpAffine(warp_img, M, (output_w, output_h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
        else:
            warped_img = cv2.warpPerspective(warp_img, M, (output_w, output_h))

        warped_base = np.zeros_like(warped_img)
        y_end = min(roi_y + base_h, output_h)
        x_end = min(roi_x + base_w, output_w)
        actual_h = y_end - roi_y
        actual_w = x_end - roi_x

        if actual_h > 0 and actual_w > 0:
            warped_base[roi_y:y_end, roi_x:x_end] = base_img[:actual_h, :actual_w]
        details['timing']['warping_ms'] = (time.time() - t0) * 1000

        mask_left = self.preprocess_for_stitching(img_left)
        mask_right = self.preprocess_for_stitching(img_right)

        # 6. 融合
        t0 = time.time()
        if direction == 1:
            img_l, img_r = warped_base, warped_img
            raw_r = img_right
        else:
            img_l, img_r = warped_img, warped_base
            raw_r = img_left

        if self.method == 'average':
            result = self._blend_average(img_l, img_r)
        elif self.method == 'laplacian':
            result = self._blend_laplacian(img_l, img_r)
        elif self.method == 'poisson':
            # 平移模式下简化Poisson融合
            if self.transform_type == TransformType.TRANSLATION:
                result = self._blend_poisson_translation(img_l, img_r)
            else:
                result = self._blend_poisson(img_l, img_r, raw_r, M)
        else:
            result = self._blend_average(img_l, img_r)

        details['timing']['fusion_ms'] = (time.time() - t0) * 1000
        details['timing']['total_ms'] = (time.time() - t_start) * 1000
        details['fusion']['method'] = self.method
        details['fusion']['output_size'] = result.shape[:2]

        if return_details:
            # 额外返回匹配可视化所需的中间数据
            details['_internal'] = {
                'kp1': kp1, 'kp2': kp2, 'matches': good_matches,
                'M': M,
                'warped_base': warped_base, 'warped_img': warped_img,
                'transform_type': self.transform_type.value
            }
            return result, details

        return result

    def _blend_average(self, img_l, img_r):
        """距离变换加权融合"""
        mask_l = (np.sum(img_l, axis=2) > 0).astype(np.float32)
        mask_r = (np.sum(img_r, axis=2) > 0).astype(np.float32)

        dist_l = cv2.distanceTransform(mask_l.astype(np.uint8), cv2.DIST_L2, 3)
        dist_r = cv2.distanceTransform(mask_r.astype(np.uint8), cv2.DIST_L2, 3)

        sum_dist = dist_l + dist_r
        sum_dist[sum_dist == 0] = 1.0

        alpha = dist_l / sum_dist
        res = np.zeros_like(img_l)
        for c in range(3):
            res[:, :, c] = img_l[:, :, c] * alpha + img_r[:, :, c] * (1 - alpha)
        return res.astype(np.uint8)

    def _blend_laplacian(self, img_l, img_r, levels=5):
        """拉普拉斯金字塔融合"""
        h, w = img_l.shape[:2]
        new_h = int(np.ceil(h / 2 ** levels) * 2 ** levels)
        new_w = int(np.ceil(w / 2 ** levels) * 2 ** levels)

        img_l = cv2.copyMakeBorder(img_l, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT)
        img_r = cv2.copyMakeBorder(img_r, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT)

        mask = np.zeros((new_h, new_w), dtype=np.float32)
        mask_l = (np.sum(img_l, axis=2) > 0)
        mask_r = (np.sum(img_r, axis=2) > 0)
        overlap = (mask_l & mask_r)
        '''
        mask[mask_l] = 1.0
        if overlap.any():
            overlap_indices = np.where(overlap)
            mid_x = int(np.median(overlap_indices[1]))
            mask[:, mid_x:] = 0.0
                '''
        if overlap.any():
            # 找到重叠区中心线作为接缝
            ys, xs = np.where(overlap)
            if len(xs) > 0:
                # 使用中位数避免异常值影响
                mid_x = int(np.median(xs))
                mask[:, :mid_x] = 1.0
            else:
                mask[mask_l] = 1.0
        else:
            mask[mask_l] = 1.0


        gp_l = [img_l.astype(np.float32)]
        gp_r = [img_r.astype(np.float32)]
        gp_m = [mask]

        for i in range(levels):
            gp_l.append(cv2.pyrDown(gp_l[-1]))
            gp_r.append(cv2.pyrDown(gp_r[-1]))
            gp_m.append(cv2.pyrDown(gp_m[-1]))

        lp_l, lp_r = [gp_l[levels]], [gp_r[levels]]
        for i in range(levels, 0, -1):
            size = (gp_l[i - 1].shape[1], gp_l[i - 1].shape[0])
            L = cv2.subtract(gp_l[i - 1], cv2.pyrUp(gp_l[i], dstsize=size))
            R = cv2.subtract(gp_r[i - 1], cv2.pyrUp(gp_r[i], dstsize=size))
            lp_l.append(L)
            lp_r.append(R)

        LS = []
        for l, r, m in zip(lp_l, lp_r, gp_m[::-1]):
            m3 = cv2.merge([m, m, m])
            LS.append(l * m3 + r * (1.0 - m3))

        res = LS[0]
        for i in range(1, levels + 1):
            size = (LS[i].shape[1], LS[i].shape[0])
            res = cv2.pyrUp(res, dstsize=size)
            res = cv2.add(res, LS[i])

        return np.clip(res[:h, :w], 0, 255).astype(np.uint8)

    def _blend_poisson(self, img_l, img_r, raw_r, full_M):
        """泊松融合"""
        mask = np.zeros(img_r.shape[:2], dtype=np.uint8)
        raw_h, raw_w = raw_r.shape[:2]
        raw_mask = np.ones((raw_h, raw_w), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(raw_mask, full_M, (img_r.shape[1], img_r.shape[0]))

        y, x = np.where(warped_mask > 0)
        if len(x) == 0:
            return self._blend_average(img_l, img_r)

        center = (int((np.min(x) + np.max(x)) / 2), int((np.min(y) + np.max(y)) / 2))

        try:
            return cv2.seamlessClone(img_r, img_l, warped_mask, center, cv2.NORMAL_CLONE)
        except:
            return self._blend_average(img_l, img_r)

    def _blend_poisson_translation(self, img_l, img_r):
        """泊松融合（平移优化版）"""
        # 找到右图的有效区域
        mask_r = (np.sum(img_r, axis=2) > 0).astype(np.uint8) * 255

        ys, xs = np.where(mask_r > 0)
        if len(xs) == 0:
            return img_l

        center = (int((np.min(xs) + np.max(xs)) / 2), int((np.min(ys) + np.max(ys)) / 2))

        try:
            return cv2.seamlessClone(img_r, img_l, mask_r, center, cv2.NORMAL_CLONE)
        except:
            return self._blend_average(img_l, img_r)
