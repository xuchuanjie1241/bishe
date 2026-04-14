
class CylinderUnwarper2:
    def unwarp(self, img, radius, focal_length):
        """
        输入：校正后的圆柱体图像, 半径, 焦距
        输出：拉平后的平面图
        """
        h, w = img.shape[:2]

        # 建立目标图（平面图）的坐标网格
        # 目标图的宽度通常比原图宽，因为圆柱侧面展开了
        # 这里简化处理，假设输出宽度与输入一致，或者根据物理模型计算
        dst_w = w
        dst_h = h

        # 生成目标图的网格坐标 (x, y)
        map_x = np.zeros((dst_h, dst_w), np.float32)
        map_y = np.zeros((dst_h, dst_w), np.float32)

        # 图像中心作为坐标原点
        center_x = w / 2
        center_y = h / 2

        # 核心映射公式：
        # 平面图上的 x_dest 对应圆柱表面的弧长
        # 对应的角度 theta = x_dest / f (或 x_dest / r，取决于投影模型，这里用柱面投影通用公式)
        # 原图坐标 x_src = f * tan(theta)  <-- 这是相机成像模型
        # 或者简化物理模型：x_src = r * sin(x_dest / r)

        # 这里使用一种适合柱状标签的逆变换：
        # 我们遍历目标图的每一个点 (y, x)，找到它在原图中的位置
        for y in range(dst_h):
            for x in range(dst_w):
                # 归一化 x 坐标 (-1 到 1 之间，相对于半径)
                # 实际物理距离 / 半径 = sin(theta)
                # 所以 x_src_offset = radius * sin( x_dest_offset / radius )

                x_dest_offset = x - center_x

                # 这是一个几何近似，假设是将圆柱皮剥下来的过程
                # 目标图上的距离是弧长，原图上的距离是投影弦长
                # 弧长 = r * theta -> theta = 弧长 / r
                # 原图 x = r * sin(theta)

                theta = x_dest_offset / radius

                # 限制 theta 范围防止越界 (-pi/2 到 pi/2)
                if -np.pi / 2 < theta < np.pi / 2:
                    x_src = radius * np.sin(theta) + center_x
                    map_x[y, x] = x_src
                    map_y[y, x] = y  # Y轴高度不变
                else:
                    map_x[y, x] = -1  # 标记为无效
                    map_y[y, x] = -1

        # 使用 remap 进行插值变换
        unwarped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        return unwarped


class CylinderUnwarper3:
    def unwarp(self, img_rect, r, f):
        """
        r: 瓶身半径(px), f: 估计焦距(px)
        手机拍摄建议 f 取 img.shape[1] * 0.8 到 1.2 之间
        """
        h, w = img_rect.shape[:2]
        # 创建映射表
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)

        center_x = w / 2
        center_y = h / 2

        for y in range(h):
            for x in range(w):
                # 核心逻辑：论文中的逆变换公式
                # 将平面坐标(x,y) 映射回 柱面坐标
                theta = (x - center_x) / f
                h_val = (y - center_y) / f

                # 柱面投影模型
                nx = np.sin(theta)
                ny = h_val
                nz = np.cos(theta)

                # 映射到原始图像像素
                x_orig = f * nx / nz + center_x
                y_orig = f * ny / nz + center_y

                map_x[y, x] = x_orig
                map_y[y, x] = y_orig

        # 使用双线性插值进行重采样
        flat = cv2.remap(img_rect, map_x, map_y, cv2.INTER_LINEAR)
        debug_map = cv2.normalize(map_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("Map X Debug", debug_map)
        cv2.imwrite("MapXDebug.png", debug_map)
        return flat


import cv2
import numpy as np


class RectangularLabelUnwarper:  # 重命名类，更符合功能
    def unwarp(self, img_rect, r, f):
        """
        r: 瓶身半径(px), f: 经校正比例调整后的精确焦距(px)
        目标：将圆柱面展开成平整的长方形标签
        """
        h, w = img_rect.shape[:2]

        # 1. 预先构造目标图的坐标网格
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

        center_x = w / 2
        center_y = h / 2

        # 2. 矩阵化核心逻辑
        # theta 决定横向展开
        theta = (x_coords - center_x) / f
        # h_val 决定纵向位置
        h_val = (y_coords - center_y) / f

        # --- 核心修改点 ---

        # 横向保持 tan 映射（将圆柱面拉直）
        map_x = f * np.tan(theta) + center_x

        # 纵向改为线性映射（消除透视，实现长方形）
        # 这里直接让 y 轴坐标与 y_coords 成正比，不随 theta 变化
        map_y = f * h_val + center_y

        # --- 修改结束 ---

        # 3. 类型转换 (remap 要求 float32)
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        # 4. 使用双线性插值进行重采样
        # 增加边界处理，超出部分设为黑色
        flat = cv2.remap(img_rect, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return flat


import cv2
import numpy as np


class CylinderUnwarper4:
    def __init__(self):
        pass

    def unwarp(self, img, radius, focal_length):
        """
        将圆柱形区域图片展开为平面
        :param img: CylinderLocator 裁剪后的主体图 (rectz1)
        :param radius: 圆柱半径 (通常由 Locator 计算或预设)
        :param focal_length: 修正后的焦距 (f_final)
        :return: 展开后的平面图
        """
        h, w = img.shape[:2]
        center_x = w / 2
        center_y = h / 2

        # 1. 创建目标展开图的坐标网格
        # 展开后的宽度通常由圆柱弧度决定，这里保持与原图宽度一致或略大
        outputs_w = w
        outputs_h = h

        # 创建映射表 (Remap maps)
        map_x = np.zeros((outputs_h, outputs_w), dtype=np.float32)
        map_y = np.zeros((outputs_h, outputs_w), dtype=np.float32)

        # 2. 填充映射逻辑
        # 使用向量化计算提高速度
        u = np.arange(outputs_w)
        v = np.arange(outputs_h)
        uu, vv = np.meshgrid(u, v)

        # 计算相对于中心的偏移
        delta_u = uu - center_x
        delta_v = vv - center_y

        # 核心投影变换
        # theta 是点在圆柱上的水平角度
        theta = delta_u / focal_length
        # h_rel 是相对于中心的高度缩放系数
        h_rel = delta_v / focal_length

        # 映射回原图坐标
        # x' = f * tan(theta) + cx
        # y' = f * h_rel / cos(theta) + cy
        res_x = focal_length * np.tan(theta) + center_x
        res_y = focal_length * h_rel / np.cos(theta) + center_y

        map_x = res_x.astype(np.float32)
        map_y = res_y.astype(np.float32)

        # 3. 执行重采样 (使用双线性插值)
        unwarped = cv2.remap(img, map_x, map_y,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)

        # 4. 自动裁剪无效黑色边缘 (可选)
        # 展开后边缘会因为坐标越界变黑，可以根据有效范围进行裁剪
        return unwarped


