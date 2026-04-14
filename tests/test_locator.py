import cv2
import numpy as np
import os
import json
from datetime import datetime
import sys

from matplotlib import pyplot as plt

from tests.utils.base_tester import BaseTester

sys.path.append('..')  # 添加父目录到路径

from modules.locator import CylinderLocator
from utils.metrics import MetricsCalculator
from utils.visualization import TestVisualizer


class LocatorTesterV2(BaseTester):
    def __init__(self):
        super().__init__('locator')

    def test_edge_detection_variants(self, img_path):
        """对比不同边缘检测方法"""
        img = cv2.imread(img_path)

        methods = [
            ('canny_adaptive', {'edge_method': 'canny', 'threshold_mode': 'adaptive'}),
            ('canny_fixed', {'edge_method': 'canny', 'threshold_mode': 'fixed'}),
            ('sobel', {'edge_method': 'sobel'})
        ]

        results = []
        for name, params in methods:
            locator = CylinderLocator()
            for k, v in params.items():
                setattr(locator, k, v)

            rect, r, details = locator.process(img, normalize=False, return_details=True)

            # 保存边缘图
            edge_vis = details['intermediate']['edges']
            self.save_artifact(f'edges_{name}.png', edge_vis, 'edges')

            results.append({
                'method': name,
                'radius': r,
                'time_ms': details['timing']['total_ms'],
                'bounds': details['bounds']
            })

        return self.log_result('edge_detection_comparison', results)

    def test_projection_accuracy(self, img_path, ground_truth_bounds):
        """测试边界定位精度"""
        img = cv2.imread(img_path)
        locator = CylinderLocator()

        rect, r, details = locator.process(img, return_details=True)

        # 计算与真值的IoU
        pred = details['bounds']
        gt = ground_truth_bounds

        # 简化的IoU计算（水平方向）
        inter_left = max(pred['left'], gt['left'])
        inter_right = min(pred['right'], gt['right'])
        intersection = max(0, inter_right - inter_left)

        union = (pred['right'] - pred['left']) + (gt['right'] - gt['left']) - intersection
        iou = intersection / union if union > 0 else 0

        # 可视化投影曲线
        proj_x = details['boundary_stats']['projection_x']
        fig, ax = plt.subplots()
        ax.plot(proj_x)
        ax.axvline(pred['left'], color='r', linestyle='--', label='Detected')
        ax.axvline(pred['right'], color='r', linestyle='--')
        ax.axvline(gt['left'], color='g', linestyle=':', label='Ground Truth')
        ax.axvline(gt['right'], color='g', linestyle=':')
        ax.legend()
        self.save_artifact('projection_x.png', fig, 'visualizations')

        return self.log_result('boundary_accuracy', {
            'iou': iou,
            'radius_error': abs(r - (gt['right'] - gt['left']) / 2)
        }, passed=iou > 0.9)
    
class LocatorTester:
    """圆柱定位模块测试类"""

    def __init__(self, output_dir='test_results/locator'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.locator = CylinderLocator()
        self.results = []

    def test_radius_accuracy(self, test_images, ground_truth_radii):
        """
        测试半径检测精度
        :param test_images: 测试图像路径列表
        :param ground_truth_radii: 对应的真实半径（像素）
        """
        print("=" * 60)
        print("测试1: 半径检测精度")
        print("=" * 60)

        errors = []
        rel_errors = []

        for img_path, gt_radius in zip(test_images, ground_truth_radii):
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 处理图像
            rect, detected_radius = self.locator.process(img, normalize=False)

            # 计算误差
            error = abs(detected_radius - gt_radius)
            rel_error = error / gt_radius * 100

            errors.append(error)
            rel_errors.append(rel_error)

            print(f"图像: {os.path.basename(img_path)}")
            print(f"  检测半径: {detected_radius:.1f}px, 真实值: {gt_radius:.1f}px")
            print(f"  绝对误差: {error:.2f}px, 相对误差: {rel_error:.2f}%")

            # 可视化边界
            h, w = img.shape[:2]
            left = int(w / 2 - detected_radius)
            right = int(w / 2 + detected_radius)
            top = int(h * 0.1)
            bottom = int(h * 0.9)

            vis_path = os.path.join(self.output_dir,
                                    f"bounds_{os.path.basename(img_path)}")
            TestVisualizer.plot_roi_bounds(img, left, right, top, bottom, vis_path)

        # 统计结果
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        mean_rel = np.mean(rel_errors)

        print(f"\n统计结果:")
        print(f"  平均绝对误差: {mean_error:.2f}±{std_error:.2f} px")
        print(f"  平均相对误差: {mean_rel:.2f}%")

        # 保存结果
        result = {
            'test_name': 'radius_accuracy',
            'mean_error_px': float(mean_error),
            'std_error_px': float(std_error),
            'mean_relative_error_percent': float(mean_rel),
            'samples': len(errors),
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(result)

        # 绘制误差分布
        error_dist_path = os.path.join(self.output_dir, 'radius_error_dist.png')
        TestVisualizer.plot_error_distribution(errors, error_dist_path,
                                               "Radius Detection Error")

        return result

    def test_normalization_consistency(self, image_groups):
        """
        测试半径归一化一致性
        :param image_groups: 多组图像，每组包含同规格圆柱体不同角度拍摄的图像
        """
        print("\n" + "=" * 60)
        print("测试2: 半径归一化一致性")
        print("=" * 60)

        consistency_results = []

        for group_idx, image_paths in enumerate(image_groups):
            self.locator.ref_radius = None  # 重置基准
            radii = []

            for img_path in image_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                rect, r = self.locator.process(img, normalize=True)
                radii.append(r)
                print(f"  组{group_idx + 1}, {os.path.basename(img_path)}: "
                      f"归一化后半径={r:.1f}px")

            if len(radii) > 1:
                cv = np.std(radii) / np.mean(radii) * 100
                consistency_results.append(cv)
                print(f"  组{group_idx + 1} 变异系数: {cv:.2f}%")

        mean_cv = np.mean(consistency_results)
        print(f"\n平均变异系数: {mean_cv:.2f}% (目标<2%)")

        result = {
            'test_name': 'normalization_consistency',
            'mean_cv_percent': float(mean_cv),
            'target': '<2%',
            'passed': mean_cv < 2.0
        }
        self.results.append(result)
        return result

    def test_adaptive_threshold(self, image_sets):
        """
        测试自适应阈值鲁棒性
        :param image_sets: dict, {'normal': [...], 'overexposed': [...],
                                   'underexposed': [...]}
        """
        print("\n" + "=" * 60)
        print("测试3: 自适应阈值鲁棒性")
        print("=" * 60)

        robustness = {}

        for condition, images in image_sets.items():
            success = 0
            total = len(images)

            for img_path in images:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                try:
                    rect, r = self.locator.process(img, normalize=False)
                    # 检查检测是否合理（半径在图像宽度的20%-80%之间）
                    h, w = img.shape[:2]
                    if r > w * 0.2 and r < w * 0.8:
                        success += 1
                except Exception as e:
                    print(f"  处理失败 {img_path}: {e}")

            rate = success / total * 100
            robustness[condition] = {
                'success_rate': rate,
                'success': success,
                'total': total
            }
            print(f"  {condition}: {success}/{total} ({rate:.1f}%)")

        result = {
            'test_name': 'adaptive_threshold',
            'robustness': robustness,
            'passed': all(r['success_rate'] > 85 for r in robustness.values())
        }
        self.results.append(result)
        return result

    def generate_report(self):
        """生成测试报告"""
        report_path = os.path.join(self.output_dir, 'test_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n测试报告已保存: {report_path}")
        return self.results


# 使用示例
if __name__ == "__main__":
    tester = LocatorTester()

    # 准备测试数据路径（根据实际情况修改）
    test_data = {
        'radius_accuracy': {
            'images': ['data/calib/cyl_50mm_01.png',
                       'data/calib/cyl_60mm_01.png',
                       'data/calib/cyl_75mm_01.png'],
            'radii': [245, 294, 368]  # 对应像素半径
        },
        'normalization': [
            ['data/multi_view/view1_01.png', 'data/multi_view/view1_02.png'],
            ['data/multi_view/view2_01.png', 'data/multi_view/view2_02.png']
        ],
        'robustness': {
            'normal': ['data/light/normal_01.png', 'data/light/normal_02.png'],
            'overexposed': ['data/light/over_01.png', 'data/light/over_02.png'],
            'underexposed': ['data/light/under_01.png', 'data/light/under_02.png']
        }
    }

    # 执行测试
    if os.path.exists('data'):
        tester.test_radius_accuracy(
            test_data['radius_accuracy']['images'],
            test_data['radius_accuracy']['radii']
        )
        tester.test_normalization_consistency(test_data['normalization'])
        tester.test_adaptive_threshold(test_data['robustness'])
        tester.generate_report()
    else:
        print("请创建测试数据目录结构或修改测试数据路径")