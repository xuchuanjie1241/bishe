import cv2
import numpy as np
import os
import json
from datetime import datetime
import sys

from modules.preprocessor import DistortionCorrector

sys.path.append('..')

from modules.unwarper import CylinderUnwarper
from modules.locator import CylinderLocator
from utils.metrics import MetricsCalculator
from utils.visualization import TestVisualizer


class UnwarperTester:
    """圆柱展开模块测试类"""

    def __init__(self, output_dir='test_results/unwarper'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.unwarper = CylinderUnwarper()
        self.locator = CylinderLocator()
        self.results = []

    # 正确计算水平方向边缘压缩（垂直方向同理）
    def calculate_true_edge_compression(corners, grid_size):
        """
        corners: 检测到的角点 (N, 2)
        grid_size: (w, h) 内角点数，如 (3, 5)
        """
        w, h = grid_size
        corners = corners.reshape(h, w, 2)  # 重塑为网格

        # 计算每行的平均水平格距（每行有 w-1 个水平间隔）
        row_spacings = []
        for row in range(h):
            row_corners = corners[row]  # 该行的所有角点
            spacings = []
            for col in range(w - 1):
                dist = np.linalg.norm(row_corners[col] - row_corners[col + 1])
                spacings.append(dist)
            row_spacings.append(np.mean(spacings))

        # 边缘行 vs 中心行
        edge_top = row_spacings[0]
        edge_bottom = row_spacings[-1]
        center_mean = np.mean(row_spacings[1:-1]) if h > 2 else row_spacings[0]

        # 边缘压缩率（百分比偏差）
        compression_top = abs(edge_top - center_mean) / center_mean * 100
        compression_bottom = abs(edge_bottom - center_mean) / center_mean * 100

        return {
            'center_spacing': float(center_mean),
            'edge_top_spacing': float(edge_top),
            'edge_bottom_spacing': float(edge_bottom),
            'edge_compression_top_%': float(compression_top),
            'edge_compression_bottom_%': float(compression_bottom),
            'max_edge_compression_%': float(max(compression_top, compression_bottom))
        }

    def test_checkerboard_accuracy(self, checkerboard_images, grid_size=(4, 5),
                                   square_size=5.0):
        """
        使用棋盘格测试展开几何精度
        :param checkerboard_images: 贴在圆柱体上的棋盘格图像路径列表
        :param grid_size: 棋盘格内角点数量 (w, h)
        :param square_size: 实际方格尺寸 (mm)
        """
        print("=" * 60)
        print("测试1: 棋盘格重投影精度")
        print("=" * 60)
        corrector = DistortionCorrector('../modules/camera_params.npz')
        all_errors = []
        sample_results = []

        for idx, img_path in enumerate(checkerboard_images):
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 定位和展开
            # rect, r = self.locator.process(img, normalize=False)
            y = 1706 / 2 - (1706 - 1082)

            # 估算焦距（简化版，实际应使用标定参数）
            f = corrector.focal_length

            # 半径是图片宽度的一半
            r = img.shape[1] / 2

            unwarped = self.unwarper.unwarp(img, r, f, center_y=y)

            # 保存unwarped图像
            unwarped_path = os.path.join('../data/output/', f'unwarped_3.jpg')
            cv2.imwrite(unwarped_path, unwarped)

            # 检测棋盘格角点
            gray = cv2.cvtColor(unwarped, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

            if not ret:
                print(f"  {img_path}: 未检测到棋盘格")
                continue

            '''
            # 计算水平格距（按行分析）
corners_grid = corners.reshape(h_corners, w_corners, 2)
row_statistics = []

for row in range(h_corners):
    row_corners = corners_grid[row]
    row_spacings = []
    for col in range(w_corners - 1):
        dist = np.linalg.norm(row_corners[col] - row_corners[col + 1])
        row_spacings.append(dist)
    
    row_mean = np.mean(row_spacings)
    row_std = np.std(row_spacings)
    row_statistics.append({
        'row': row,
        'mean': float(row_mean),
        'std': float(row_std),
        'cv_%': float(row_std / row_mean * 100) if row_mean > 0 else 0
    })

# 分析边缘效应
center_rows = row_statistics[1:-1] if h_corners > 2 else row_statistics
edge_top = row_statistics[0]
edge_bottom = row_statistics[-1]
center_mean = np.mean([r['mean'] for r in center_rows])

edge_compression_top = abs(edge_top['mean'] - center_mean) / center_mean * 100
edge_compression_bottom = abs(edge_bottom['mean'] - center_mean) / center_mean * 100

print(f"  水平方向均匀性分析:")
print(f"    中心区域平均格距: {center_mean:.2f}px")
print(f"    顶边格距: {edge_top['mean']:.2f}px (偏差: {edge_compression_top:.2f}%)")
print(f"    底边格距: {edge_bottom['mean']:.2f}px (偏差: {edge_compression_bottom:.2f}%)")

# 垂直方向（参考值，不应作为误差指标）
v_spacings = []
for col in range(w_corners):
    for row in range(h_corners - 1):
        dist = np.linalg.norm(corners_grid[row, col] - corners_grid[row + 1, col])
        v_spacings.append(dist)

mean_v = np.mean(v_spacings)
anisotropy_ratio = mean_v / center_mean
print(f"  各向异性比 (V/H): {anisotropy_ratio:.2f} (1.0-2.0为正常范围)")

# 真正的质量评价
quality_score = max(edge_compression_top, edge_compression_bottom)
print(f"  边缘压缩率: {quality_score:.2f}% (目标<10%)")
            '''

            # 计算相邻角点间距
            corners = corners.reshape(-1, 2)
            horizontal_distances = []
            vertical_distances = []

            for i in range(grid_size[1]):  # 行
                for j in range(grid_size[0] - 1):  # 列-1
                    idx1 = i * grid_size[0] + j
                    idx2 = idx1 + 1
                    dist = np.linalg.norm(corners[idx1] - corners[idx2])
                    horizontal_distances.append(dist)

            for i in range(grid_size[1] - 1):  # 行-1
                for j in range(grid_size[0]):  # 列
                    idx1 = i * grid_size[0] + j
                    idx2 = idx1 + grid_size[0]
                    dist = np.linalg.norm(corners[idx1] - corners[idx2])
                    vertical_distances.append(dist)

            # 计算方格尺寸的像素值与一致性
            mean_h = np.mean(horizontal_distances)
            std_h = np.std(horizontal_distances)
            mean_v = np.mean(vertical_distances)
            std_v = np.std(vertical_distances)


            print(f"图像 {idx + 1}:")
            print(f"  水平格距: {mean_h:.2f}±{std_h:.2f} px")
            print(f"  垂直格距: {mean_v:.2f}±{std_h:.2f} px")

            all_errors.extend([abs(d - mean_h) for d in horizontal_distances])

            # 可视化
            vis_img = unwarped.copy()
            cv2.drawChessboardCorners(vis_img, grid_size, corners, ret)
            cv2.imwrite(os.path.join(self.output_dir,
                                     f'checkerboard_3.png'), vis_img)

            sample_results.append({
                'image': img_path,
                'mean_grid_px': float(mean_h),
                'grid_consistency_cv': float(std_h / mean_h * 100),
            })

        mean_error, std_error, max_error = MetricsCalculator.reprojection_error(
            np.array([[0, 0]]), np.array([[0, 0]])  # 占位，实际使用all_errors
        )
        mean_error = np.mean(all_errors)

        print(f"\n总体统计:")
        print(f"  平均格距偏差: {mean_error:.2f} px")
        print(f"  最大格距偏差: {np.max(all_errors):.2f} px")

        result = {
            'test_name': 'checkerboard_accuracy',
            'mean_grid_error_px': float(mean_error),
            'max_grid_error_px': float(np.max(all_errors)),
            'samples': sample_results,
            'passed': mean_error < 2.0  # 阈值2像素
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


if __name__ == "__main__":
    tester = UnwarperTester()

    # 示例测试数据
    if os.path.exists('data'):
        tester.test_checkerboard_accuracy(['data/checkerboard/cyl_checker_03.jpg'])
        tester.test_line_straightness(['data/stripes/cyl_stripes_01.png'])
        tester.test_focal_length_sensitivity(
            'data/test/sample.png',
            r=300,
            f_values=[0.8, 0.9, 1.0, 1.1, 1.2]
        )
        tester.generate_report()
    else:
        print("请准备测试数据")