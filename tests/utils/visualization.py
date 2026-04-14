import cv2
import numpy as np
import matplotlib.pyplot as plt


class TestVisualizer:
    """测试结果可视化"""

    @staticmethod
    def save_comparison_figure(images, titles, save_path, figsize=(15, 5)):
        """保存对比图"""
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]

        for i, (img, title) in enumerate(zip(images, titles)):
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            axes[i].set_title(title)
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def draw_matches(img1, kp1, img2, kp2, matches, save_path, max_matches=50):
        """绘制特征匹配结果"""
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2,
            matches[:max_matches], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(save_path, img_matches)
        return img_matches

    @staticmethod
    def plot_error_distribution(errors, save_path, title="Error Distribution"):
        """绘制误差分布直方图"""
        plt.figure(figsize=(8, 5))
        plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Error (pixels)')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.axvline(np.mean(errors), color='r', linestyle='--',
                    label=f'Mean: {np.mean(errors):.2f}')
        plt.legend()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_roi_bounds(img, left, right, top, bottom, save_path):
        """可视化检测到的边界"""
        vis_img = img.copy()
        h, w = img.shape[:2]
        # 绘制边界线
        cv2.line(vis_img, (left, 0), (left, h), (0, 255, 0), 2)
        cv2.line(vis_img, (right, 0), (right, h), (0, 255, 0), 2)
        cv2.line(vis_img, (0, top), (w, top), (255, 0, 0), 2)
        cv2.line(vis_img, (0, bottom), (w, bottom), (255, 0, 0), 2)

        # 绘制中心点
        cx, cy = (left + right) // 2, (top + bottom) // 2
        cv2.circle(vis_img, (cx, cy), 5, (0, 0, 255), -1)

        cv2.imwrite(save_path, vis_img)
        return vis_img