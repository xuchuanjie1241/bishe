# test_stitcher.py (配套修改版)
import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
import sys

sys.path.append('..')

from modules.stitcher import SimpleStitcher
from utils.metrics import MetricsCalculator
from utils.visualization import TestVisualizer


class StitcherTester:
    """拼接融合模块测试类"""

    def __init__(self, output_dir='test_results/stitcher'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'matches'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'blending'), exist_ok=True)
        self.results = []

    def test_feature_matching_accuracy(self, image_pairs):
        """
        测试特征匹配精度，对比不同策略
        :param image_pairs: [(img1_path, img2_path), ...]
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

            # 为每个策略创建stitcher并测试
            for strategy in strategies:
                stitcher = SimpleStitcher(method='average', match_strategy=strategy)

                # 使用return_details获取详细信息
                start_time = time.time()
                result, details = stitcher.stitch(img1, img2, direction=1, return_details=True)
                elapsed = (time.time() - start_time) * 1000

                if result is not None:
                    # 提取匹配统计
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

                    # 可视化匹配（仅对'full'策略保存，避免太多文件）
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

        # 汇总统计
        summary = self._summarize_matching(matching_results)

        result = {
            'test_name': 'feature_matching_accuracy',
            'summary': summary,
            'details': matching_results,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(result)

        print(f"\n策略对比汇总:")
        print(f"  平均内点率 - Lowe: {summary['avg_inlier_ratio']['lowe']:.2f}, "
              f"Vertical: {summary['avg_inlier_ratio']['vertical']:.2f}, "
              f"RANSAC: {summary['avg_inlier_ratio']['ransac']:.2f}, "
              f"Full: {summary['avg_inlier_ratio']['full']:.2f}")

        return result

    def test_blending_quality(self, image_pairs_with_conditions):
        """
        测试不同融合算法质量
        :param image_pairs_with_conditions: [(img1, img2, condition_name), ...]
                                          condition如: 'normal', 'illumination_diff', 'high_contrast'
        """
        print("\n" + "=" * 60)
        print("测试2: 融合算法质量对比")
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

            # 先用full策略获取对齐，再测试不同融合方法
            align_stitcher = SimpleStitcher(method='average', match_strategy='full')
            _, align_details = align_stitcher.stitch(img1, img2, return_details=True)

            if align_details is None or '_internal' not in align_details:
                print("  对齐失败，跳过此测试对")
                continue

            # 获取对齐后的图像对（用于公平比较融合算法）
            warped_base = align_details['_internal']['warped_base']
            warped_img = align_details['_internal']['warped_img']
            M = align_details['_internal']['M']

            for method in methods:
                print(f"  测试 {method} 融合...", end=" ")

                try:
                    # 直接调用融合方法，跳过重匹配
                    stitcher = SimpleStitcher(method=method)

                    t0 = time.time()
                    if method == 'average':
                        fused = stitcher._blend_average(warped_base, warped_img)
                    elif method == 'laplacian':
                        fused = stitcher._blend_laplacian(warped_base, warped_img)
                    elif method == 'poisson':
                        # Poisson需要原图信息，这里简化处理
                        fused = stitcher._blend_poisson(warped_base, warped_img,
                                                        img2, M)
                    elapsed = (time.time() - t0) * 1000

                    # 保存结果
                    output_path = os.path.join(
                        self.output_dir, 'blending',
                        f'{condition}_{method}_pair{idx}.png'
                    )
                    cv2.imwrite(output_path, fused)

                    # 计算质量指标
                    # 1. SSIM: 与两幅输入图像的结构相似性（取平均）
                    # 注意：由于几何变换，这里简化处理，仅评估融合区域的平滑性

                    # 2. 接缝可见度指数 (Seam Visibility Index)
                    # 找到接缝位置（重叠区中心）
                    mask_base = (np.sum(warped_base, axis=2) > 0)
                    mask_img = (np.sum(warped_img, axis=2) > 0)
                    overlap = mask_base & mask_img

                    if np.any(overlap):
                        # 计算重叠区质心作为接缝位置
                        ys, xs = np.where(overlap)
                        seam_x = int(np.median(xs))
                        svi = MetricsCalculator.seam_visibility_index(fused, seam_x)
                    else:
                        svi = 1.0

                    # 3. 亮度连续性：计算重叠区左右亮度差异
                    if np.any(overlap):
                        left_edge = fused[:, max(0, seam_x - 10):seam_x]
                        right_edge = fused[:, seam_x:min(fused.shape[1], seam_x + 10)]
                        mean_diff = np.abs(np.mean(left_edge) - np.mean(right_edge))
                    else:
                        mean_diff = 0



                    pair_result['methods'][method] = {
                        'success': True,
                        'ssim_vs_input': None,  # 需要原始对齐图计算，这里省略
                        'seam_visibility_index': round(float(svi), 4),
                        'intensity_discontinuity': round(float(mean_diff), 2),
                        'time_ms': round(elapsed, 2),
                        'output_path': output_path
                    }

                    print(f"SVI={svi:.3f}, 亮度差={mean_diff:.1f}, 耗时={elapsed:.1f}ms")

                except Exception as e:
                    print(f"失败: {e}")
                    pair_result['methods'][method] = {
                        'success': False,
                        'error': str(e)
                    }

            quality_results.append(pair_result)

        # 生成对比表格
        self._print_blending_comparison(quality_results)

        result = {
            'test_name': 'blending_quality',
            'details': quality_results,
            'recommendation': self._recommend_blending_method(quality_results)
        }
        self.results.append(result)
        return result

    def test_processing_speed(self, image_pairs, iterations=5):
        """
        基准速度测试
        """
        print("\n" + "=" * 60)
        print("测试3: 处理速度基准")
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

            for img1_path, img2_path in image_pairs[:3]:  # 取前3对平均
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
            'platform': 'CPU',  # 可修改为实际平台
            'iterations': iterations
        }
        self.results.append(result)
        return result

    def test_geometric_accuracy(self, image_pairs_with_ground_truth):
        """
        几何配准精度测试（需要真值Homography矩阵）
        :param image_pairs_with_ground_truth: [(img1, img2, H_ground_truth), ...]
        """
        print("\n" + "=" * 60)
        print("测试4: 几何配准精度")
        print("=" * 60)

        accuracy_results = []

        for idx, (img1_path, img2_path, H_gt) in enumerate(image_pairs_with_ground_truth):
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            stitcher = SimpleStitcher(match_strategy='full')
            _, details = stitcher.stitch(img1, img2, return_details=True)

            if details is None or 'M' not in details.get('_internal', {}):
                continue

            H_est = details['_internal']['M']

            # 计算矩阵差异（Frobenius范数）
            # 注意：H可能相差一个尺度因子，需要归一化
            H_est_norm = H_est / H_est[2, 2]
            H_gt_norm = H_gt / H_gt[2, 2]

            diff = np.linalg.norm(H_est_norm - H_gt_norm, 'fro')

            # 计算重投影误差（使用4个角点）
            h, w = img2.shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

            corners_est = cv2.perspectiveTransform(corners, H_est)
            corners_gt = cv2.perspectiveTransform(corners, H_gt)

            reproj_error, _, max_error = MetricsCalculator.reprojection_error(
                corners_est.reshape(-1, 2), corners_gt.reshape(-1, 2)
            )

            accuracy_results.append({
                'pair_id': idx,
                'matrix_frobenius_diff': round(float(diff), 4),
                'reprojection_error_mean': round(float(reproj_error), 2),
                'reprojection_error_max': round(float(max_error), 2)
            })

            print(f"  图像对 {idx + 1}: 矩阵差={diff:.4f}, "
                  f"重投影误差={reproj_error:.2f}px (最大{max_error:.2f}px)")

        mean_error = np.mean([r['reprojection_error_mean'] for r in accuracy_results])
        print(f"\n平均重投影误差: {mean_error:.2f}px")

        result = {
            'test_name': 'geometric_accuracy',
            'details': accuracy_results,
            'mean_reprojection_error': round(float(mean_error), 2),
            'passed': mean_error < 5.0  # 5像素阈值
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

    def _print_blending_comparison(self, results):
        """打印融合对比表格"""
        print("\n融合质量对比表:")
        print("-" * 90)
        print(f"{'场景':<20} {'方法':<12} {'SVI':>8} {'亮度差':>10} {'耗时(ms)':>10} {'状态':>6}")
        print("-" * 90)

        for res in results:
            condition = res['condition']
            for method, metrics in res['methods'].items():
                if metrics.get('success'):
                    print(f"{condition:<20} {method:<12} "
                          f"{metrics['seam_visibility_index']:>8.3f} "
                          f"{metrics['intensity_discontinuity']:>10.1f} "
                          f"{metrics['time_ms']:>10.1f} {'OK':>6}")
                else:
                    print(f"{condition:<20} {method:<12} {'FAIL':>8} "
                          f"{'-':>10} {'-':>10} {'FAIL':>6}")

    def _recommend_blending_method(self, results):
        """基于测试结果推荐融合方法"""
        scores = {'average': 0, 'laplacian': 0, 'poisson': 0}

        for res in results:
            for method, metrics in res['methods'].items():
                if not metrics.get('success'):
                    continue
                # SVI越接近1越好
                svi_score = 1.0 / abs(metrics['seam_visibility_index'] - 1.0 + 0.01)
                # 亮度差越小越好
                int_score = 1.0 / (metrics['intensity_discontinuity'] + 1.0)
                # 时间越快越好（但权重较低）
                time_score = 1000.0 / (metrics['time_ms'] + 1.0)

                scores[method] += svi_score * 0.5 + int_score * 0.3 + time_score * 0.2

        best = max(scores, key=scores.get)
        return {
            'recommended': best,
            'scores': scores,
            'reason': '基于接缝可见度、亮度连续性和计算效率的综合评分'
        }

    def generate_report(self):
        """生成完整测试报告"""
        report_path = os.path.join(self.output_dir, 'stitcher_test_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

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
    ]

    if os.path.exists('data/pair1'):
        # 1. 特征匹配策略测试
        tester.test_feature_matching_accuracy(test_pairs)

        # 2. 融合质量测试（包含不同光照条件）
        blending_tests = [
            (test_pairs[0][0], test_pairs[0][1], 'normal'),
            (test_pairs[1][0], test_pairs[1][1], 'illumination_diff')
            if len(test_pairs) > 1 else (test_pairs[0][0], test_pairs[0][1], 'normal')
        ]
        tester.test_blending_quality(blending_tests)

        # 3. 速度测试
        tester.test_processing_speed(test_pairs, iterations=3)

        # 生成报告
        tester.generate_report()
    else:
        print("请准备测试数据或修改测试路径")
        print("期望的数据结构：")
        print("  data/pair1/left.jpg, right.jpg")
        print("  data/pair2/left.jpg, right.jpg")