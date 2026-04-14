import cv2
import numpy as np
import os
import json
import time
import sys

sys.path.append('..')

from modules.locator import CylinderLocator
from modules.unwarper import CylinderUnwarper
from modules.stitcher import SimpleStitcher
from utils.metrics import MetricsCalculator


class IntegrationTester:
    """端到端集成测试"""

    def __init__(self, output_dir='test_results/integration'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.locator = CylinderLocator()
        self.unwarper = CylinderUnwarper()
        self.stitcher = SimpleStitcher(method='laplacian')
        self.results = []

    def test_full_pipeline(self, image_sequence):
        """
        测试完整流程
        :param image_sequence: 按顺序排列的图像路径列表（3-6张）
        """
        print("=" * 60)
        print("集成测试: 完整流程")
        print("=" * 60)

        if len(image_sequence) < 2:
            print("至少需要2张图像")
            return

        # 模拟main.py的处理流程
        timings = {
            'locate': [],
            'unwarp': [],
            'stitch': []
        }

        flats = []

        # 阶段1: 预处理和展开
        print("阶段1: 圆柱定位与展开...")
        self.locator.set_reference(None)  # 重置

        for idx, img_path in enumerate(image_sequence):
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 定位
            t0 = time.time()
            if idx == 0:
                rect, r = self.locator.process(img, normalize=False)
                self.locator.set_reference(r)
            else:
                rect, r = self.locator.process(img, normalize=True)
            t1 = time.time()
            timings['locate'].append((t1 - t0) * 1000)

            # 展开
            f = rect.shape[1] * 1.5
            t0 = time.time()
            flat = self.unwarper.unwarp(rect, r, f)
            t1 = time.time()
            timings['unwarp'].append((t1 - t0) * 1000)

            flats.append(flat)
            print(f"  图像{idx + 1}: 定位{timings['locate'][-1]:.1f}ms, "
                  f"展开{timings['unwarp'][-1]:.1f}ms")

        # 阶段2: 拼接（中心扩展策略）
        print("\n阶段2: 图像拼接...")

        # 从中间向两侧拼接
        center = len(flats) // 2
        panorama = flats[center]

        # 向右拼接
        for i in range(center + 1, len(flats)):
            t0 = time.time()
            panorama = self.stitcher.stitch(panorama, flats[i], direction=1)
            t1 = time.time()
            timings['stitch'].append((t1 - t0) * 1000)
            print(f"  向右拼接 {i}: {timings['stitch'][-1]:.1f}ms")

        # 向左拼接
        for i in range(center - 1, -1, -1):
            t0 = time.time()
            panorama = self.stitcher.stitch(flats[i], panorama, direction=-1)
            t1 = time.time()
            timings['stitch'].append((t1 - t0) * 1000)
            print(f"  向左拼接 {i}: {timings['stitch'][-1]:.1f}ms")

        # 保存结果
        result_path = os.path.join(self.output_dir, 'panorama_result.png')
        cv2.imwrite(result_path, panorama)

        # 统计
        total_time = sum(timings['locate']) + sum(timings['unwarp']) + sum(timings['stitch'])

        print(f"\n处理统计:")
        print(f"  图像数量: {len(flats)}")
        print(f"  平均定位时间: {np.mean(timings['locate']):.1f}ms")
        print(f"  平均展开时间: {np.mean(timings['unwarp']):.1f}ms")
        print(f"  平均拼接时间: {np.mean(timings['stitch']):.1f}ms")
        print(f"  总耗时: {total_time:.1f}ms")
        print(f"  输出分辨率: {panorama.shape[1]}x{panorama.shape[0]}")

        result = {
            'test_name': 'full_pipeline',
            'image_count': len(flats),
            'timings_ms': {
                'locate_mean': float(np.mean(timings['locate'])),
                'unwarp_mean': float(np.mean(timings['unwarp'])),
                'stitch_mean': float(np.mean(timings['stitch'])),
                'total': float(total_time)
            },
            'output_resolution': [int(panorama.shape[1]), int(panorama.shape[0])],
            'output_path': result_path
        }
        self.results.append(result)
        return result

    def test_error_accumulation(self, image_sequence):
        """
        测试误差累积效应（对比级联 vs 中心扩展）
        """
        print("\n" + "=" * 60)
        print("测试: 误差累积对比")
        print("=" * 60)

        # 准备展开图
        flats = []
        self.locator.set_reference(None)

        for img_path in image_sequence:
            img = cv2.imread(img_path)
            rect, r = self.locator.process(img, normalize=False if len(flats) == 0 else True)
            f = rect.shape[1] * 1.5
            flat = self.unwarper.unwarp(rect, r, f)
            flats.append(flat)

        # 策略1: 级联拼接（0->1->2->3）
        print("策略1: 级联拼接")
        cascade = flats[0]
        for i in range(1, len(flats)):
            cascade = self.stitcher.stitch(cascade, flats[i], direction=1)

        # 策略2: 中心扩展（已在上一个测试中实现）
        print("策略2: 中心扩展拼接")
        center = len(flats) // 2
        center_expand = flats[center]
        for i in range(center + 1, len(flats)):
            center_expand = self.stitcher.stitch(center_expand, flats[i], direction=1)
        for i in range(center - 1, -1, -1):
            center_expand = self.stitcher.stitch(flats[i], center_expand, direction=-1)

        # 保存对比
        cv2.imwrite(os.path.join(self.output_dir, 'cascade.png'), cascade)
        cv2.imwrite(os.path.join(self.output_dir, 'center_expand.png'), center_expand)

        # 评估：计算最终全景图的梯度连续性（间接反映对齐质量）
        def evaluate_smoothness(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            return np.mean(np.abs(grad_x))

        smooth_cascade = evaluate_smoothness(cascade)
        smooth_center = evaluate_smoothness(center_expand)

        print(f"级联拼接平滑度: {smooth_cascade:.2f}")
        print(f"中心扩展平滑度: {smooth_center:.2f}")
        print(f"改进幅度: {(smooth_center - smooth_cascade) / smooth_cascade * 100:.1f}%")

        result = {
            'test_name': 'error_accumulation',
            'cascade_smoothness': float(smooth_cascade),
            'center_expand_smoothness': float(smooth_center),
            'improvement_percent': float((smooth_center - smooth_cascade) / smooth_cascade * 100)
        }
        self.results.append(result)
        return result

    def test_full_pipeline_detailed(self, image_sequence):
        """详细的端到端测试"""
        timeline = []  # 记录每个阶段的时间和参数

        for idx, img_path in enumerate(image_sequence):
            step = {'image_idx': idx, 'stages': {}}

            # Locator
            t0 = time.time()
            locator = CylinderLocator()
            rect, r, det_loc = locator.process(
                cv2.imread(img_path),
                return_details=True
            )
            step['stages']['locate'] = {
                'time_ms': det_loc['timing']['total_ms'],
                'radius': r,
                'bounds': det_loc['bounds']
            }

            # Unwarper
            f = rect.shape[1] * 1.5
            unwarped, det_unw = self.unwarper.unwarp(rect, r, f, return_details=True)
            step['stages']['unwarp'] = {
                'time_ms': det_unw['timing']['total_ms'],
                'geometry': det_unw['geometry'],
                'deformation': det_unw['deformation_analysis']
            }

            timeline.append(step)

        # 可以生成甘特图展示各阶段耗时...
        return timeline

    def generate_report(self):
        """生成测试报告"""
        report_path = os.path.join(self.output_dir, 'integration_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n集成测试报告已保存: {report_path}")
        return self.results


if __name__ == "__main__":
    tester = IntegrationTester()

    if os.path.exists('data/sequence'):
        # 获取序列图像
        seq_dir = 'data/sequence'
        images = sorted([os.path.join(seq_dir, f)
                         for f in os.listdir(seq_dir)
                         if f.endswith(('.png', '.jpg'))])

        if len(images) >= 3:
            tester.test_full_pipeline(images[:4])  # 测试前4张
            tester.test_error_accumulation(images[:4])
            tester.generate_report()
        else:
            print("需要至少3张序列图像")
    else:
        print("请准备序列测试数据")