import cv2
import numpy as np


def generate_checkerboard(rows=9, cols=7, square_size=100, padding=100):
    """
    生成棋盘格图像
    :param rows: 内角点行数 + 1 (即格子的行数)
    :param cols: 内角点列数 + 1 (即格子的列数)
    :param square_size: 每个格子的像素宽度
    :param padding: 四周留白的像素宽度
    """
    # 计算图像总尺寸
    width = cols * square_size + 2 * padding
    height = rows * square_size + 2 * padding

    # 创建纯白背景
    img = np.ones((height, width), dtype=np.uint8) * 255

    for r in range(rows):
        for c in range(cols):
            # 如果行列之和为偶数，涂黑（国际象棋布局）
            if (r + c) % 2 == 0:
                y = padding + r * square_size
                x = padding + c * square_size
                img[y:y + square_size, x:x + square_size] = 0

    return img


# 生成一个 8x6 的棋盘格 (内角点为 7x5)
board = generate_checkerboard(rows=6, cols=8, square_size=150)
cv2.imwrite('checkerboard_pattern.png', board)
print("棋盘格已生成：checkerboard_pattern.png")