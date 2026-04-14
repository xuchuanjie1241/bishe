import cv2
import numpy as np
import os
import json
import sys

sys.path.append('..')

from modules.locator import CylinderLocator
from modules.unwarper import CylinderUnwarper
from modules.stitcher import SimpleStitcher


class RobustnessTester:
    """鲁棒性测试"""

    def __init__(self, output_dir='test_results/robustness'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []

    def test_jitter_robustness(self, base_image, jitter_angles):
        """
        测试抖动容限
        :param jitter_angles: [5, 10, 15] 度
        """
        print("=" * 60)
        print("鲁棒性测试: 拍摄抖动")
        print("=" * 60)

        img = cv2.imread(base_image)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        locator = CylinderLocator()
        unwarper = CylinderUnwarper()

        results = []

        for angle in jitter_angles:
            # 模拟旋转抖动
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(128, 128, 128))

            try:
                rect, r = locator.process(rotated, normalize=False)
                f = rect.shape[1] * 1.5
                unwarped = unwarper.unwarp(rect, r, f)

                # 检查展开质量（通过边缘检测判断是否有明显畸变）
                gray = cv2.cvtColor(unwarped, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_ratio = np.sum(edges > 0) / (unwarped.shape[0] * unwarped.shape[1])

                success = edge_ratio > 0.01  # 简单阈值判断

                results.append({
                    'angle': angle,
                    'success': success,
                    'edge_ratio': float(edge_ratio),
                    'output_size': unwarped.shape[:2]
                })

                cv2.imwrite(os.path.join(self.output_dir, f'jitter_{angle}deg.png'),
                            unwarped)

                print(f"  抖动{angle}°: {'成功' if success else '失败'} "
                      f"(边缘占比: {edge_ratio:.3f})")

            except Exception as e:
                results.append({
                    'angle': angle,
                    'success': False,
                    'error': str(e)
                })
                print(f"  抖动{angle}°: 失败 - {e}")

        success_rate = sum(1 for r in results if r['success']) / len(results) * 100
        print(f"\n成功率: {success_rate:.0f}%")

        result = {
            'test_name': 'jitter_robustness',
            'angles_tested': jitter_angles,
            'results': results,
            'success_rate': float(success_rate)
        }
        self.results.append(result)
        return result

    def test_illumination_robustness(self, image_path):
        """
        测试光照变化鲁棒性
        """
        print("\n" + "=" * 60)
        print("鲁棒性测试: 光照变化")
        print("=" * 60)

        img = cv2.imread(image_path)
        locator = CylinderLocator()

        variations = {
            'original': img,
            'bright': cv2.convertScaleAbs(img, alpha=1.3, beta=30),
            'dark': cv2.convertScaleAbs(img, alpha=0.7, beta=-30),
            'contrast_low': cv2.convertScaleAbs(img, alpha=0.5, beta=64)
        }

        results = {}

        for name, variant in variations.items():
            try:
                rect, r = locator.process(variant, normalize=False)
                success = True
                radius = float(r)
            except Exception as e:
                success = False
                radius = 0

            results[name] = {
                'success': success,
                'radius_px': radius
            }

            status = "成功" if success else "失败"
            print(f"  {name}: {status}, 半径: {radius:.1f}px")

            # 保存可视化结果
            if success:
                cv2.imwrite(os.path.join(self.output_dir, f'illum_{name}.png'), rect)

        # 检查半径一致性
        radii = [r['radius_px'] for r in results.values() if r['success']]
        if len(radii) > 1:
            cv_radius = np.std(radii) / np.mean(radii) * 100
            print(f"半径一致性 (CV): {cv_radius:.2f}%")

        result = {
            'test_name': 'illumination',
            'variations': results,
            'radius_cv_percent': float(cv_radius) if len(radii) > 1 else None
        }
        self.results.append(result)
        return result

    def generate_report(self):
        report_path = os.path.join(self.output_dir, 'robustness_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n鲁棒性测试报告已保存: {report_path}")
        return self.results


if __name__ == "__main__":
    tester = RobustnessTester()

    if os.path.exists('data/test_sample.png'):
        tester.test_jitter_robustness('data/test_sample.png', [5, 10, 15])
        tester.test_illumination_robustness('data/test_sample.png')
        tester.generate_report()
    else:
        print("请准备测试样本")