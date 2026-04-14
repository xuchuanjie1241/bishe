# test_stitcher.py (重写版 - 集成先进融合指标与变换模型对比)
import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
import sys
from typing import Dict, List, Tuple, Optional
import warnings

# 添加父目录到路径
sys.path.append('..')
sys.path.append('.')

from modules.stitcher import SimpleStitcher, TransformType
from utils.metrics import MetricsCalculator

# 尝试导入skimage，若未安装则给出警告
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.filters import sobel
    from skimage.util import img_as_float

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not installed. Some metrics will use fallback implementations.")


class AdvancedMetricsCalculator:
    """高级图像融合质量指标计算器"""

    @staticmethod
    def ms_ssim(img1: np.ndarray, img2: np.ndarray,
                scales: int = 5, weights: Optional[np.ndarray] = None) -> float:
        """
        多尺度结构相似性指数 (MS-SSIM)
        :param img1: 参考图像
        :param img2: 待评估图像
        :param scales: 尺度数量
        :param weights: 各尺度权重，默认 [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        :return: MS-SSIM 值 (0-1, 越接近1越好)
        """
        if weights is None:
            weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        weights = weights[:scales]
        weights /= np.sum(weights)

        # 转为灰度图并归一化到0-1
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        img1 = img1.astype(np.float64) / 255.0
        img2 = img2.astype(np.float64) / 255.0

        mssim_values = []
        for i in range(scales):
            # 计算当前尺度的SSIM
            if SKIMAGE_AVAILABLE:
                ssim_val = ssim(img1, img2, data_range=1.0)
            else:
                ssim_val = AdvancedMetricsCalculator._compute_ssim(img1, img2)
            mssim_values.append(ssim_val)

            # 下采样
            if i < scales - 1:
                img1 = cv2.pyrDown(img1)
                img2 = cv2.pyrDown(img2)

        return np.prod(np.power(mssim_values, weights))

    @staticmethod
    def gmsd(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        梯度幅值相似性偏差 (GMSD)
        参考文献: Gradient Magnitude Similarity Deviation (Zhang et al., 2013)
        :return: GMSD 值 (越小越好，理想值为0)
        """
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        # 计算梯度幅值 (使用Sobel算子)
        sobelx1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
        sobely1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
        gm1 = np.sqrt(sobelx1 ** 2 + sobely1 ** 2)

        sobelx2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
        sobely2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
        gm2 = np.sqrt(sobelx2 ** 2 + sobely2 ** 2)

        # 梯度幅值相似性
        c = 0.0026  # 常数，基于灰度范围0-255
        gm_sim = (2 * gm1 * gm2 + c) / (gm1 ** 2 + gm2 ** 2 + c)

        # 标准差作为GMSD
        gmsd_val = np.std(gm_sim)
        return gmsd_val

    @staticmethod
    def qabf(img_fused: np.ndarray, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        基于梯度的融合质量指标 (QAB/F)
        参考文献: A Fusion Performance Measure (Xydeas & Petrovic, 2000)
        :param img_fused: 融合图像
        :param img1: 源图像1 (左图)
        :param img2: 源图像2 (右图)
        :return: QABF 值 (0-1, 越接近1越好)
        """

        def compute_edge_info(img):
            """计算边缘强度和方向"""
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float64)

            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

            g = np.sqrt(sobelx ** 2 + sobely ** 2)
            a = np.arctan2(sobely, sobelx)
            return g, a

        # 计算各图像的梯度信息
        g_f, a_f = compute_edge_info(img_fused)
        g_1, a_1 = compute_edge_info(img1)
        g_2, a_2 = compute_edge_info(img2)

        # 相对边缘强度保留值
        def edge_preservation(g_source, g_fused):
            """边缘保留率"""
            mask = g_source > g_fused
            g_ratio = np.zeros_like(g_source)
            g_ratio[mask] = g_fused[mask] / (g_source[mask] + 1e-10)
            g_ratio[~mask] = g_source[~mask] / (g_fused[~mask] + 1e-10)
            return g_ratio

        g_1f = edge_preservation(g_1, g_f)
        g_2f = edge_preservation(g_2, g_f)

        # 边缘方向一致性
        a_diff_1 = np.abs(np.abs(a_f - a_1) - np.pi / 2) / (np.pi / 2)
        a_diff_2 = np.abs(np.abs(a_f - a_2) - np.pi / 2) / (np.pi / 2)

        # 计算权重 (基于边缘强度)
        w_1 = g_1 / (g_1 + g_2 + 1e-10)
        w_2 = g_2 / (g_1 + g_2 + 1e-10)

        # QAB/F 计算
        q_1 = g_1f * (1 - a_diff_1)
        q_2 = g_2f * (1 - a_diff_2)

        qabf = np.sum(w_1 * q_1 + w_2 * q_2) / (img_fused.shape[0] * img_fused.shape[1])
        return float(qabf)

    @staticmethod
    def vif(img1: np.ndarray, img2: np.ndarray, sigma_nsq: float = 2.0) -> float:
        """
        视觉信息保真度 (VIF)
        简化的VIF实现，基于高斯尺度混合模型
        :param sigma_nsq: 噪声方差估计
        :return: VIF 值 (0-1, 越接近1越好)
        """
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        # 使用滑动窗口计算局部统计量
        window_size = 8
        stride = window_size // 2

        num = 0.0
        den = 0.0

        for i in range(0, img1.shape[0] - window_size, stride):
            for j in range(0, img1.shape[1] - window_size, stride):
                window1 = img1[i:i + window_size, j:j + window_size]
                window2 = img2[i:i + window_size, j:j + window_size]

                mu1 = np.mean(window1)
                mu2 = np.mean(window2)

                sigma1_sq = np.var(window1)
                sigma2_sq = np.var(window2)
                sigma12 = np.mean((window1 - mu1) * (window2 - mu2))

                # VIF分量计算
                g = sigma12 / (sigma1_sq + 1e-10)
                sv_sq = sigma2_sq - g * sigma12

                num += np.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq))
                den += np.log10(1 + sigma1_sq / sigma_nsq)

        if den == 0:
            return 0.0
        return num / den

    @staticmethod
    def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """基础SSIM实现 (备用)"""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(np.mean(ssim_map))


class StitcherTester:
    """拼接融合模块测试类 (增强版)"""

    def __init__(self, output_dir='test_results/stitcher'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'matches'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'blending'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'transform_comparison'), exist_ok=True)
        self.results = []
        self.metrics_calc = AdvancedMetricsCalculator()

    def test_feature_matching_accuracy(self, image_pairs):
        """
        测试特征匹配精度，对比不同策略
        """
        print("=" * 60)
        print("测试1: 特征匹配策略对比")
        print("=" * 60)

        strategies = ['lowe', 'vertical', 'ransac', 'full']
        matching_results = []

        for idx, (img1_path, img2_path) in enumerate(image_pairs):
            print(f"\n图像对 {idx + 1}: {os.path.basename(img1_path)} + {os.path.basename(img2_path)}")

            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            if img1 is None or img2 is None:
                print("  跳过：无法读取图像")
                continue

            pair_result = {
                'pair_id': idx,
                'image1': os.path.basename(img1_path),
                'image2': os.path.basename(img2_path),
                'strategies': {}
            }

            for strategy in strategies:
                stitcher = SimpleStitcher(method='average', match_strategy=strategy)

                start_time = time.time()
                result, details = stitcher.stitch(img1, img2, direction=1, return_details=True)
                elapsed = (time.time() - start_time) * 1000

                if result is not None:
                    match_stats = details['matching']
                    transform_stats = details['transform']

                    pair_result['strategies'][strategy] = {
                        'success': True,
                        'time_ms': round(elapsed, 2),
                        'keypoints_left': match_stats.get('keypoints_left', 0),
                        'keypoints_right': match_stats.get('keypoints_right', 0),
                        'initial_matches': match_stats.get('initial_matches', 0),
                        'after_lowe': match_stats.get('after_lowe', 0),
                        'after_vertical': match_stats.get('after_vertical', 0),
                        'after_ransac': match_stats.get('after_ransac', 0),
                        'final_matches': match_stats.get('final_matches', 0),
                        'inliers': transform_stats.get('inliers', 0),
                        'outliers': transform_stats.get('outliers', 0),
                        'inlier_ratio': round(transform_stats.get('inliers', 0) /
                                              max(match_stats.get('final_matches', 1), 1), 3)
                    }

                    if strategy == 'full' and 'matches' in details.get('_internal', {}):
                        self._save_match_visualization(
                            img1, img2,
                            details['_internal']['kp1'],
                            details['_internal']['kp2'],
                            details['_internal']['matches'],
                            f"pair{idx}_{strategy}.png"
                        )
                else:
                    pair_result['strategies'][strategy] = {
                        'success': False,
                        'time_ms': round(elapsed, 2),
                        'error': 'Stitching failed'
                    }

                stats = pair_result['strategies'][strategy]
                status = "✓" if stats['success'] else "✗"
                print(f"  [{status}] {strategy:10s}: "
                      f"匹配点={stats.get('final_matches', 0):3d}, "
                      f"内点率={stats.get('inlier_ratio', 0):.2f}, "
                      f"耗时={stats['time_ms']:.1f}ms")

            matching_results.append(pair_result)

        summary = self._summarize_matching(matching_results)

        result = {
            'test_name': 'feature_matching_accuracy',
            'summary': summary,
            'details': matching_results,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(result)

        print(f"\n策略对比汇总:")
        for strategy in ['lowe', 'vertical', 'ransac', 'full']:
            ratio = summary['avg_inlier_ratio'].get(strategy, 0)
            print(f"  {strategy:10s}: 平均内点率={ratio:.2f}")

        return result

    def test_blending_quality(self, image_pairs_with_conditions):
        """
        测试不同融合算法质量 (使用MS-SSIM, GMSD, QABF, VIF)
        """
        print("\n" + "=" * 60)
        print("测试2: 融合算法质量对比 (先进指标)")
        print("=" * 60)

        methods = ['average', 'laplacian', 'poisson']
        quality_results = []

        for idx, (img1_path, img2_path, condition) in enumerate(image_pairs_with_conditions):
            print(f"\n测试场景 {idx + 1}: {condition}")

            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            if img1 is None or img2 is None:
                continue

            pair_result = {
                'pair_id': idx,
                'condition': condition,
                'methods': {}
            }

            # 使用FULL策略对齐，确保几何一致性
            align_stitcher = SimpleStitcher(method='average', match_strategy='full')
            _, align_details = align_stitcher.stitch(img1, img2, return_details=True)

            if align_details is None or '_internal' not in align_details:
                print("  对齐失败，跳过此测试对")
                continue

            warped_base = align_details['_internal']['warped_base']
            warped_img = align_details['_internal']['warped_img']
            M = align_details['_internal']['M']

            for method in methods:
                print(f"  测试 {method} 融合...", end=" ")

                try:
                    stitcher = SimpleStitcher(method=method)

                    t0 = time.time()
                    if method == 'average':
                        fused = stitcher._blend_average(warped_base, warped_img)
                    elif method == 'laplacian':
                        fused = stitcher._blend_laplacian(warped_base, warped_img)
                    elif method == 'poisson':
                        # 泊松融合需要原始图像信息
                        if align_details['transform']['matrix_type'] == 'Translation':
                            fused = stitcher._blend_poisson_translation(warped_base, warped_img)
                        else:
                            fused = stitcher._blend_poisson(warped_base, warped_img,
                                                            img2, M)
                    elapsed = (time.time() - t0) * 1000

                    # 保存结果
                    output_path = os.path.join(
                        self.output_dir, 'blending',
                        f'{condition}_{method}_pair{idx}.png'
                    )
                    cv2.imwrite(output_path, fused)

                    # 计算先进质量指标
                    metrics = {}

                    # 1. MS-SSIM (与左图和右图分别计算，取平均)
                    try:
                        mssim_left = self.metrics_calc.ms_ssim(warped_base, fused)
                        mssim_right = self.metrics_calc.ms_ssim(warped_img, fused)
                        metrics['ms_ssim'] = round((mssim_left + mssim_right) / 2, 4)
                    except Exception as e:
                        metrics['ms_ssim'] = None
                        print(f"[MS-SSIM Error: {e}]")

                    # 2. GMSD (越小越好)
                    try:
                        gmsd_left = self.metrics_calc.gmsd(warped_base, fused)
                        gmsd_right = self.metrics_calc.gmsd(warped_img, fused)
                        metrics['gmsd'] = round((gmsd_left + gmsd_right) / 2, 4)
                    except Exception as e:
                        metrics['gmsd'] = None
                        print(f"[GMSD Error: {e}]")

                    # 3. QAB/F (基于梯度的融合质量)
                    try:
                        # 需要找到重叠区域的有效部分进行计算
                        h, w = fused.shape[:2]
                        # 裁剪到有效区域避免黑边影响
                        gray_fused = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
                        mask = (gray_fused > 0).astype(np.uint8)
                        ys, xs = np.where(mask)
                        if len(xs) > 0:
                            x1, x2 = max(0, xs.min()), min(w, xs.max())
                            y1, y2 = max(0, ys.min()), min(h, ys.max())

                            valid_fused = fused[y1:y2, x1:x2]
                            valid_base = warped_base[y1:y2, x1:x2]
                            valid_img = warped_img[y1:y2, x1:x2]

                            qabf_val = self.metrics_calc.qabf(valid_fused, valid_base, valid_img)
                            metrics['qabf'] = round(qabf_val, 4)
                        else:
                            metrics['qabf'] = 0.0
                    except Exception as e:
                        metrics['qabf'] = None
                        print(f"[QABF Error: {e}]")

                    # 4. VIF (视觉信息保真度)
                    try:
                        vif_left = self.metrics_calc.vif(warped_base, fused)
                        vif_right = self.metrics_calc.vif(warped_img, fused)
                        metrics['vif'] = round((vif_left + vif_right) / 2, 4)
                    except Exception as e:
                        metrics['vif'] = None
                        print(f"[VIF Error: {e}]")

                    pair_result['methods'][method] = {
                        'success': True,
                        'metrics': metrics,
                        'time_ms': round(elapsed, 2),
                        'output_path': output_path
                    }

                    print(f"MS-SSIM={metrics.get('ms_ssim', 0):.3f}, "
                          f"GMSD={metrics.get('gmsd', 0):.3f}, "
                          f"QAB/F={metrics.get('qabf', 0):.3f}, "
                          f"VIF={metrics.get('vif', 0):.3f}, "
                          f"耗时={elapsed:.1f}ms")

                except Exception as e:
                    print(f"失败: {e}")
                    pair_result['methods'][method] = {
                        'success': False,
                        'error': str(e)
                    }

            quality_results.append(pair_result)

        self._print_blending_comparison_advanced(quality_results)

        result = {
            'test_name': 'blending_quality_advanced',
            'details': quality_results,
            'recommendation': self._recommend_blending_method_advanced(quality_results)
        }
        self.results.append(result)
        return result

    def test_transform_comparison(self, image_pairs):
        """
        对比不同几何变换模型的性能 (HOMOGRAPHY vs AFFINE vs TRANSLATION)
        """
        print("\n" + "=" * 60)
        print("测试3: 几何变换模型对比 (HOMOGRAPHY vs AFFINE vs TRANSLATION)")
        print("=" * 60)

        transform_types = [
            (TransformType.HOMOGRAPHY, "单应性变换(8DoF)"),
            (TransformType.AFFINE, "仿射变换(6DoF)"),
            (TransformType.TRANSLATION, "纯平移(2DoF)")
        ]

        comparison_results = []

        for idx, (img1_path, img2_path) in enumerate(image_pairs):
            print(f"\n图像对 {idx + 1}: {os.path.basename(img1_path)} + {os.path.basename(img2_path)}")

            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            if img1 is None or img2 is None:
                continue

            pair_result = {
                'pair_id': idx,
                'image1': os.path.basename(img1_path),
                'image2': os.path.basename(img2_path),
                'transforms': {}
            }

            for transform_type, type_name in transform_types:
                print(f"  测试 {type_name}...", end=" ")

                try:
                    stitcher = SimpleStitcher(
                        method='laplacian',  # 使用效果较好的融合方法
                        match_strategy='full',
                        transform_type=transform_type
                    )

                    t0 = time.time()
                    result, details = stitcher.stitch(img1, img2, direction=1, return_details=True)
                    elapsed = (time.time() - t0) * 1000

                    if result is not None and details is not None:
                        # 提取变换参数
                        transform_stats = details.get('transform', {})
                        matching_stats = details.get('matching', {})

                        # 计算重投影误差（使用匹配点）
                        internal = details.get('_internal', {})
                        if 'M' in internal and 'matches' in internal:
                            M = internal['M']
                            kp1 = internal['kp1']
                            kp2 = internal['kp2']
                            matches = internal['matches']

                            # 计算几何误差
                            src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

                            if transform_type == TransformType.HOMOGRAPHY:
                                projected = cv2.perspectiveTransform(src_pts, M)
                            elif transform_type == TransformType.AFFINE:
                                projected = cv2.transform(src_pts, M[:2, :])
                            else:  # TRANSLATION
                                projected = src_pts + np.array([M[0, 2], M[1, 2]])

                            errors = np.linalg.norm(projected - dst_pts, axis=2).flatten()
                            mean_error = np.mean(errors)
                            max_error = np.max(errors)
                            std_error = np.std(errors)
                        else:
                            mean_error = max_error = std_error = -1

                        pair_result['transforms'][transform_type.value] = {
                            'success': True,
                            'time_ms': round(elapsed, 2),
                            'inliers': transform_stats.get('inliers', 0),
                            'outliers': transform_stats.get('outliers', 0),
                            'reproj_error_mean': round(float(mean_error), 2),
                            'reproj_error_max': round(float(max_error), 2),
                            'reproj_error_std': round(float(std_error), 2),
                            'final_matches': matching_stats.get('final_matches', 0),
                            'transform_params': transform_stats.get('transform_params', {})
                        }

                        # 保存结果图
                        output_path = os.path.join(
                            self.output_dir, 'transform_comparison',
                            f'pair{idx}_{transform_type.value}.png'
                        )
                        cv2.imwrite(output_path, result)

                        print(f"内点={pair_result['transforms'][transform_type.value]['inliers']}, "
                              f"重投影误差={mean_error:.2f}px, "
                              f"耗时={elapsed:.1f}ms")
                    else:
                        pair_result['transforms'][transform_type.value] = {
                            'success': False,
                            'error': 'Stitching failed'
                        }
                        print("失败")

                except Exception as e:
                    print(f"异常: {e}")
                    pair_result['transforms'][transform_type.value] = {
                        'success': False,
                        'error': str(e)
                    }

            comparison_results.append(pair_result)

        # 汇总分析
        self._summarize_transform_comparison(comparison_results)

        result = {
            'test_name': 'transform_comparison',
            'details': comparison_results,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(result)
        return result

    def test_processing_speed(self, image_pairs, iterations=5):
        """
        基准速度测试
        """
        print("\n" + "=" * 60)
        print("测试4: 处理速度基准")
        print("=" * 60)

        speed_results = {}

        for method in ['average', 'laplacian', 'poisson']:
            print(f"\n测试方法: {method}")
            times = {
                'feature_extraction': [],
                'matching': [],
                'transform_estimation': [],
                'warping': [],
                'fusion': [],
                'total': []
            }

            stitcher = SimpleStitcher(method=method, match_strategy='full')

            for img1_path, img2_path in image_pairs[:3]:
                img1 = cv2.imread(img1_path)
                img2 = cv2.imread(img2_path)

                if img1 is None or img2 is None:
                    continue

                # 预热
                _ = stitcher.stitch(img1, img2, return_details=True)

                # 正式测试
                for _ in range(iterations):
                    _, details = stitcher.stitch(img1, img2, return_details=True)
                    if details:
                        t = details['timing']
                        times['feature_extraction'].append(t.get('feature_extraction_ms', 0))
                        times['matching'].append(t.get('matching_ms', 0))
                        times['transform_estimation'].append(t.get('transform_estimation_ms', 0))
                        times['warping'].append(t.get('warping_ms', 0))
                        times['fusion'].append(t.get('fusion_ms', 0))
                        times['total'].append(t.get('total_ms', 0))

            # 计算平均
            avg_times = {k: round(np.mean(v), 2) for k, v in times.items() if v}
            speed_results[method] = avg_times

            print(f"  特征提取: {avg_times.get('feature_extraction', 0):.1f}ms")
            print(f"  特征匹配: {avg_times.get('matching', 0):.1f}ms")
            print(f"  变换估计: {avg_times.get('transform_estimation', 0):.1f}ms")
            print(f"  图像变换: {avg_times.get('warping', 0):.1f}ms")
            print(f"  图像融合: {avg_times.get('fusion', 0):.1f}ms")
            print(f"  总计: {avg_times.get('total', 0):.1f}ms "
                  f"({1000 / avg_times.get('total', 1):.1f} FPS)")

        result = {
            'test_name': 'processing_speed',
            'results': speed_results,
            'platform': 'CPU',
            'iterations': iterations
        }
        self.results.append(result)
        return result

    def _save_match_visualization(self, img1, img2, kp1, kp2, matches, filename):
        """保存匹配可视化图"""
        match_img = cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0),
            singlePointColor=(0, 0, 255)
        )
        path = os.path.join(self.output_dir, 'matches', filename)
        cv2.imwrite(path, match_img)

    def _summarize_matching(self, results):
        """汇总匹配统计"""
        summary = {
            'total_pairs': len(results),
            'strategies': {},
            'avg_inlier_ratio': {}
        }

        for strategy in ['lowe', 'vertical', 'ransac', 'full']:
            ratios = []
            final_matches = []
            times = []

            for pair in results:
                if strategy in pair['strategies'] and pair['strategies'][strategy]['success']:
                    s = pair['strategies'][strategy]
                    ratios.append(s.get('inlier_ratio', 0))
                    final_matches.append(s.get('final_matches', 0))
                    times.append(s['time_ms'])

            summary['avg_inlier_ratio'][strategy] = round(np.mean(ratios), 3) if ratios else 0
            summary['strategies'][strategy] = {
                'avg_matches': round(np.mean(final_matches), 1) if final_matches else 0,
                'avg_time_ms': round(np.mean(times), 1) if times else 0
            }

        return summary

    def _print_blending_comparison_advanced(self, results):
        """打印融合对比表格 (先进指标版)"""
        print("\n融合质量对比表 (MS-SSIM/GMSD/QAB/F/VIF):")
        print("-" * 110)
        print(
            f"{'场景':<20} {'方法':<12} {'MS-SSIM':>10} {'GMSD':>10} {'QAB/F':>10} {'VIF':>10} {'耗时(ms)':>10} {'状态':>6}")
        print("-" * 110)

        for res in results:
            condition = res['condition']
            for method, metrics in res['methods'].items():
                if metrics.get('success'):
                    m = metrics['metrics']
                    print(f"{condition:<20} {method:<12} "
                          f"{m.get('ms_ssim', 0):>10.3f} "
                          f"{m.get('gmsd', 0):>10.3f} "
                          f"{m.get('qabf', 0):>10.3f} "
                          f"{m.get('vif', 0):>10.3f} "
                          f"{metrics['time_ms']:>10.1f} {'OK':>6}")
                else:
                    print(f"{condition:<20} {method:<12} {'FAIL':>10} "
                          f"{'FAIL':>10} {'FAIL':>10} {'FAIL':>10} "
                          f"{'-':>10} {'FAIL':>6}")

    def _recommend_blending_method_advanced(self, results):
        """基于先进指标推荐融合方法"""
        scores = {'average': 0, 'laplacian': 0, 'poisson': 0}

        for res in results:
            for method, metrics in res['methods'].items():
                if not metrics.get('success'):
                    continue
                m = metrics['metrics']

                # 归一化评分 (越高越好)
                # MS-SSIM: 直接值 (0-1)
                ssim_score = m.get('ms_ssim', 0)
                # GMSD: 转换为得分 (越小越好)
                gmsd_score = 1.0 / (1.0 + m.get('gmsd', 1))
                # QAB/F: 直接值 (0-1)
                qabf_score = m.get('qabf', 0)
                # VIF: 直接值 (0-1)
                vif_score = m.get('vif', 0)
                # 时间效率
                time_score = 1000.0 / (metrics['time_ms'] + 100)

                # 加权综合 (质量权重更高)
                total_score = (ssim_score * 0.3 +
                               gmsd_score * 0.25 +
                               qabf_score * 0.25 +
                               vif_score * 0.1 +
                               time_score * 0.1)

                scores[method] += total_score

        best = max(scores, key=scores.get)
        return {
            'recommended': best,
            'scores': {k: round(v, 3) for k, v in scores.items()},
            'reason': '基于MS-SSIM(30%), GMSD(25%), QAB/F(25%), VIF(10%), 效率(10%)的综合评分'
        }

    def _summarize_transform_comparison(self, results):
        """汇总变换模型对比结果"""
        print("\n变换模型对比汇总:")
        print("-" * 80)
        print(f"{'变换类型':<20} {'成功率':>10} {'平均内点':>10} {'平均重投影误差':>15} {'平均耗时':>10}")
        print("-" * 80)

        summary = {
            'homography': {'success': 0, 'total': 0, 'inliers': [], 'errors': [], 'times': []},
            'affine': {'success': 0, 'total': 0, 'inliers': [], 'errors': [], 'times': []},
            'translation': {'success': 0, 'total': 0, 'inliers': [], 'errors': [], 'times': []}
        }

        for pair in results:
            for ttype, data in pair['transforms'].items():
                summary[ttype]['total'] += 1
                if data.get('success'):
                    summary[ttype]['success'] += 1
                    summary[ttype]['inliers'].append(data.get('inliers', 0))
                    summary[ttype]['errors'].append(data.get('reproj_error_mean', 0))
                    summary[ttype]['times'].append(data.get('time_ms', 0))

        for ttype, data in summary.items():
            if data['total'] > 0:
                success_rate = data['success'] / data['total']
                avg_inliers = np.mean(data['inliers']) if data['inliers'] else 0
                avg_error = np.mean(data['errors']) if data['errors'] else 0
                avg_time = np.mean(data['times']) if data['times'] else 0

                print(f"{ttype:<20} {success_rate:>9.1%} {avg_inliers:>10.1f} "
                      f"{avg_error:>15.2f}px {avg_time:>10.1f}ms")

        print("-" * 80)
        print("注: 对于圆柱面展开图像，AFFINE通常比HOMOGRAPHY更稳定，TRANSLATION最快但精度受限")

    def generate_report(self):
        """生成完整测试报告"""
        report_path = os.path.join(self.output_dir, 'stitcher_test_report_advanced.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n详细测试报告已保存: {report_path}")

        # 生成摘要
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.results),
            'passed': sum(1 for r in self.results if r.get('passed', True)),
            'summary_by_test': {r['test_name']: r.get('passed', True)
                                for r in self.results}
        }

        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return self.results


# 使用示例
if __name__ == "__main__":
    tester = StitcherTester()

    # 准备测试数据（请根据实际路径修改）
    test_pairs = [
        ('data/pair1/left.jpg', 'data/pair1/right.jpg'),
        ('data/pair2/left.jpg', 'data/pair2/right.jpg'),
        ('data/pair3/left.jpg', 'data/pair3/right.jpg'),
    ]

    if os.path.exists('data/pair1'):
        print("开始测试流程...")

        # 1. 特征匹配策略测试
        tester.test_feature_matching_accuracy(test_pairs)

        # 2. 几何变换模型对比 (新增)
        tester.test_transform_comparison(test_pairs[:2])  # 取前2对进行变换对比

        # 3. 融合质量测试 (使用MS-SSIM, GMSD, QABF, VIF)
        blending_tests = [
            (test_pairs[0][0], test_pairs[0][1], 'normal'),
            (test_pairs[1][0], test_pairs[1][1], 'illumination_diff'),
            (test_pairs[0][0], test_pairs[0][1], 'high_contrast')
        ]
        tester.test_blending_quality(blending_tests)

        # 4. 速度测试
        tester.test_processing_speed(test_pairs, iterations=3)

        # 生成报告
        tester.generate_report()

        print("\n测试完成!")
    else:
        print("请准备测试数据或修改测试路径")
        print("期望的数据结构：")
        print("  data/pair1/left.jpg, right.jpg")
        print("  data/pair2/left.jpg, right.jpg")