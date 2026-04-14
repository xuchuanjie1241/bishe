import cv2
import numpy as np
class SimpleStitcher2:
    def stitch(self, img_left, img_right):
        # 1. 转灰度
        g1 = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # 2. 特征提取 (根据OpenCV版本调整，新版无需xfeatures2d)
        try:
            sift = cv2.SIFT_create()
        except AttributeError:
            sift = cv2.xfeatures2d.SIFT_create()

        kp1, des1 = sift.detectAndCompute(g1, None)
        kp2, des2 = sift.detectAndCompute(g2, None)

        # 3. 匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []

        # 4. 筛选与约束 (导师重点：垂直拉伸规律一致 -> y坐标偏差极小)
        # 阈值设定：允许大约 image height 1% - 2% 的垂直误差，视具体图片质量而定
        y_tolerance = img_left.shape[0] * 0.02

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                # 获取特征点坐标
                pt1 = kp1[m.queryIdx].pt  # img_left 的点
                pt2 = kp2[m.trainIdx].pt  # img_right 的点

                # [关键约束]：由于是圆柱面展开后的平移拼接，
                # 同一个特征点在左右两图中的 Y 坐标应该几乎一致（或有固定的微小整体偏移）
                if abs(pt1[1] - pt2[1]) < y_tolerance:
                    good.append(m)

        if len(good) < 10:
            print("匹配点不足")
            return None

        # 提取坐标点
        # 注意：我们要把 img_right (dst) 变换对齐到 img_left (src)
        # 所以计算矩阵时，src 是 right_pts, dst 是 left_pts
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

        # 5. 计算变换矩阵
        # 使用 estimateAffinePartial2D 而非 findHomography
        # 原因：你的场景限制为"平移与缩放"，AffinePartial 限制了只有 旋转、缩放、平移 (4个自由度)
        # 这比 Homography (8个自由度) 更不容易在无特征区域产生扭曲
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

        if M is None:
            print("无法计算变换矩阵")
            return None

        # 6. 计算画布大小 (Warping Canvas)
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]

        # 获取 img_right 变换后的四个角点
        corners_right = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        # 对于仿射变换，用 transform 而不是 perspectiveTransform
        transformed_corners = cv2.transform(corners_right, M)

        # 获取 img_left 的角点
        corners_left = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

        # 合并所有点以计算整体边界
        all_points = np.concatenate((corners_left, transformed_corners), axis=0)

        [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

        # 计算平移修正量 (如果变换后的图跑到了负坐标区，需要移回来)
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        # 构造最终的 3x3 变换矩阵 (用于 warpPerspective)
        # 即使是仿射变换，为了方便融合，通常扩展为 3x3 进行 warpPerspective 处理
        M_extended = np.vstack([M, [0, 0, 1]])
        full_transform = H_translation.dot(M_extended)

        # 7. 图像融合
        output_w = x_max - x_min
        output_h = y_max - y_min

        # 将 img_right 变换到画布
        warped_right = cv2.warpPerspective(img_right, full_transform, (output_w, output_h))

        # 将 img_left 放入画布 (需要加上平移修正量)
        # 创建一个基底
        canvas = np.zeros_like(warped_right)
        # 定义左图在画布上的ROI
        roi_x = translation_dist[0]
        roi_y = translation_dist[1]

        # 简单的图层叠加 (Mask处理)
        # 此处使用简单的最大值法融合，适合标签拼接（避免鬼影），
        # 如果需要更平滑的过渡，可使用多频段融合或线性加权
        canvas[roi_y:roi_y + h1, roi_x:roi_x + w1] = img_left

        # 创建掩膜以处理重叠区域
        mask_right = cv2.warpPerspective(np.ones((h2, w2), dtype=np.uint8) * 255, full_transform, (output_w, output_h))
        mask_left = np.zeros((output_h, output_w), dtype=np.uint8)
        mask_left[roi_y:roi_y + h1, roi_x:roi_x + w1] = 255

        # 融合逻辑：
        # 1. 只有左图区域 -> 左图
        # 2. 只有右图区域 -> 右图
        # 3. 重叠区域 -> 简单平均 或 线性渐变 (这里演示简单替换，优先保留变形较小的img_left)

        # 利用布尔索引进行融合
        # 只要 warped_right 有像素的地方，就先放 warped_right
        final_img = warped_right.copy()
        # 在左图存在的区域，直接覆盖左图 (因为左图是基准，未变形，清晰度最高)
        final_img[roi_y:roi_y + h1, roi_x:roi_x + w1] = img_left

        # *优化：如果想要去除明显拼接缝，可以使用 cv2.addWeighted 在重叠区做混合*
        # 但考虑到是标签拼接，文字的清晰度通常比平滑更重要，直接覆盖通常效果更好。

        return final_img


class SimpleStitcher3:
    def __init__(self):
        # 初始化 SIFT
        try:
            self.sift = cv2.SIFT_create()
        except AttributeError:
            self.sift = cv2.xfeatures2d.SIFT_create()

        # 使用 FLANN 匹配器，对手机拍摄的复杂背景更鲁棒
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def stitch(self, img_left, img_right):
        # 1. 预处理
        g1 = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # 2. 特征点提取
        kp1, des1 = self.sift.detectAndCompute(g1, None)
        kp2, des2 = self.sift.detectAndCompute(g2, None)

        if des1 is None or des2 is None:
            return None

        # 3. 匹配与初步筛选 (Lowe's Ratio Test)
        matches = self.flann.knnMatch(des1, des2, k=2)

        # 4. 结合导师建议的【垂直约束】与【运动一致性约束】
        good = []
        # 手机拍摄允许 2% 的垂直误差
        y_tolerance = img_left.shape[0] * 0.02

        # 计算所有匹配点的平均垂直位移，排除背景杂点的随机干扰
        y_offsets = [kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1] for m, n in matches if
                     m.distance < 0.75 * n.distance]
        if not y_offsets: return None
        avg_y_offset = np.median(y_offsets)  # 使用中位数排除异常值

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt

                # 约束1：单点垂直偏差在容差内
                # 约束2：该点的偏差与全局平均偏差接近（防止透明背景干扰）
                current_y_offset = pt1[1] - pt2[1]
                if abs(current_y_offset - avg_y_offset) < y_tolerance:
                    good.append(m)

        if len(good) < 10:
            print(f"匹配点不足: {len(good)}")
            return None

        # 5. 计算稳健的变换矩阵
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

        # 针对圆柱展开图，使用仿射变换防止产生非物理的“扭曲挤压”
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if M is None: return None

        # 6. 画布构建与变换
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]

        # 计算变换后的角点以确定新画布尺寸
        corners_right = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.transform(corners_right, M)

        # 结合左图，求最大边界
        corners_left = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        all_points = np.concatenate((corners_left, transformed_corners), axis=0)

        x_min, y_min = np.int32(all_points.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(all_points.max(axis=0).ravel() + 0.5)

        # 构造平移矩阵修正负坐标
        trans_x, trans_y = -x_min, -y_min
        T = np.array([[1, 0, trans_x], [0, 1, trans_y], [0, 0, 1]], dtype=np.float32)

        # 融合仿射矩阵到 3x3 空间
        M_3x3 = np.vstack([M, [0, 0, 1]])
        full_M = T.dot(M_3x3)

        # 7. 渐进式融合 (Alpha Blending) 处理透明/非正对光影
        output_w, output_h = x_max - x_min, y_max - y_min
        warped_right = cv2.warpPerspective(img_right, full_M, (output_w, output_h))

        # 创建画布并放置左图
        canvas = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        canvas[trans_y:trans_y + h1, trans_x:trans_x + w1] = img_left

        # 生成遮罩进行平滑融合
        mask_l = np.zeros((output_h, output_w), dtype=np.float32)
        mask_l[trans_y:trans_y + h1, trans_x:trans_x + w1] = 1.0

        mask_r = cv2.warpPerspective(np.ones((h2, w2), dtype=np.float32), full_M, (output_w, output_h))

        # 重叠区域进行线性融合，减少“非正对”带来的亮度差
        overlap = (mask_l > 0) & (mask_r > 0)
        res = canvas.astype(np.float32)

        # 在重叠区执行渐变
        # 这里为了简单采用 0.5 固定融合，若追求极致可使用距离变换做渐变
        res[overlap] = canvas[overlap] * 0.5 + warped_right[overlap] * 0.5
        # 非重叠区直接保留
        res[~overlap & (mask_r > 0)] = warped_right[~overlap & (mask_r > 0)]

        return res.astype(np.uint8)



class SimpleStitcher3:
    def __init__(self, method='laplacian'):
        self.method = method
        self.y_alignment_window = 0.5  # 使用图像中央 15% 区域进行 Y 对齐估计

    def estimate_vertical_offset(self, img_left, img_right):
        """估计两图之间的垂直偏移（基于重叠区域的特征点）"""
        h, w = img_left.shape[:2]

        # 提取重叠区域（左图右半边 vs 右图左半边）
        overlap_w = int(w * self.y_alignment_window)
        left_roi = img_left[:, -overlap_w:]  # 左图右侧
        right_roi = img_right[:, :overlap_w]  # 右图左侧

        # 转为灰度
        g_left = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
        g_right = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)

        # 使用 SIFT 或 ORB 快速匹配
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(g_left, None)
        kp2, des2 = sift.detectAndCompute(g_right, None)

        if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
            return 0.0  # 无法估计，假设无偏移

        # 快速匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 3:
            return 0.0

        # 计算 Y 偏移的中位数（鲁棒估计）
        y_offsets = []
        for m in good:
            pt_left = kp1[m.queryIdx].pt
            pt_right = kp2[m.trainIdx].pt
            # 注意坐标系：pt_left 在左图 ROI 内，需要加上 (w - overlap_w) 才是原图坐标
            # 但计算偏移时只需相对差
            y_offsets.append(pt_left[1] - pt_right[1])

        return np.median(y_offsets)

    def stitch(self, img_left, img_right):
        """拼接两张图，使用 estimateAffinePartial2D 保持刚体约束"""
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]

        # 1. 预处理：补偿垂直偏移（确保接缝处高度一致）
        y_offset = self.estimate_vertical_offset(img_left, img_right)
        if abs(y_offset) > 1.0:
            # 平移右图使其与左图对齐
            M_translate = np.float32([[1, 0, 0], [0, 1, -y_offset]])
            img_right = cv2.warpAffine(img_right, M_translate, (w2, h2),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0, 0, 0))

        # 2. 特征提取（全图）
        g1 = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(g1, None)
        kp2, des2 = sift.detectAndCompute(g2, None)

        if des1 is None or des2 is None:
            return None

        # 3. 匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        y_tolerance = img_left.shape[0] * 0.02

        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    # 圆柱展开后应为纯平移，Y 坐标差异应极小
                    pt1 = kp1[m.queryIdx].pt
                    pt2 = kp2[m.trainIdx].pt
                    # 经过 y_offset 补偿后，允许 2 像素容差
                    if abs(pt1[1] - pt2[1]) < y_tolerance:
                        good.append(m)

        if len(good) < 6:
            print(f"匹配点不足: {len(good)}")
            return None

        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

        # 4. 使用 estimateAffinePartial2D（刚体变换：旋转+平移+均匀缩放）
        # 这比 Homography 更适合圆柱展开图的拼接
        M, mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            refineIters=100
        )

        if M is None:
            print("无法计算变换矩阵")
            return None

        # 验证变换矩阵的合理性（防止缩放过大）
        scale_x = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
        scale_y = np.sqrt(M[1, 0] ** 2 + M[1, 1] ** 2)
        if abs(scale_x - 1.0) > 0.3 or abs(scale_y - 1.0) > 0.3:
            print(f"警告: 缩放异常 ({scale_x:.2f}, {scale_y:.2f})")

        # 5. 计算画布大小（使用 affine 变换）
        # affine 是 2x3 矩阵，需要转换为 3x3 用于透视变换计算边界
        M_3x3 = np.vstack([M, [0, 0, 1]])

        corners_right = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners_right, M_3x3)
        corners_left = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

        all_points = np.concatenate((corners_left, transformed_corners), axis=0)
        [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]],
                                  [0, 1, translation_dist[1]],
                                  [0, 0, 1]])

        # 组合变换：先 M，再平移
        M_final = H_translation.dot(M_3x3)

        output_w, output_h = x_max - x_min, y_max - y_min

        # 6. 变换右图
        warped_right = cv2.warpPerspective(img_right, M_final, (output_w, output_h))

        # 7. 放置左图
        warped_left = np.zeros_like(warped_right)
        roi_x, roi_y = translation_dist[0], translation_dist[1]

        # 确保不越界
        y_end = min(roi_y + h1, output_h)
        x_end = min(roi_x + w1, output_w)
        h_visible = y_end - roi_y
        w_visible = x_end - roi_x

        warped_left[roi_y:y_end, roi_x:x_end] = img_left[:h_visible, :w_visible]

        # 8. 融合
        if self.method == 'average':
            return self._blend_average(warped_left, warped_right)
        elif self.method == 'laplacian':
            return self._blend_laplacian(warped_left, warped_right)
        else:
            return self._blend_average(warped_left, warped_right)

    def _blend_average(self, img_l, img_r):
        """基于距离变换的羽化融合"""
        mask_l = (np.sum(img_l, axis=2) > 0).astype(np.float32)
        mask_r = (np.sum(img_r, axis=2) > 0).astype(np.float32)

        # 距离变换
        dist_l = cv2.distanceTransform((mask_l > 0).astype(np.uint8), cv2.DIST_L2, 5)
        dist_r = cv2.distanceTransform((mask_r > 0).astype(np.uint8), cv2.DIST_L2, 5)

        # 仅在重叠区域计算权重
        overlap = (mask_l > 0) & (mask_r > 0)

        alpha = np.zeros_like(mask_l)
        alpha[overlap] = dist_l[overlap] / (dist_l[overlap] + dist_r[overlap] + 1e-6)
        alpha[~overlap] = mask_l[~overlap]  # 非重叠区直接取左图

        alpha = np.clip(alpha, 0, 1)
        alpha_3 = np.stack([alpha] * 3, axis=2)

        result = img_l * alpha_3 + img_r * (1 - alpha_3)
        return result.astype(np.uint8)

    def _blend_laplacian(self, img_l, img_r, levels=4):
        """简化的拉普拉斯金字塔融合"""
        # 确保尺寸为 2^levels 的倍数
        h, w = img_l.shape[:2]
        new_h = ((h + 2 ** levels - 1) // 2 ** levels) * 2 ** levels
        new_w = ((w + 2 ** levels - 1) // 2 ** levels) * 2 ** levels

        img_l = cv2.copyMakeBorder(img_l, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT)
        img_r = cv2.copyMakeBorder(img_r, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT)

        # 生成掩码（左图在左，右图在右，中间过渡）
        mask = np.zeros((new_h, new_w), dtype=np.float32)
        # 找到重叠区域中心线
        overlap_l = np.sum(img_l, axis=2) > 0
        overlap_r = np.sum(img_r, axis=2) > 0
        overlap = overlap_l & overlap_r

        if np.any(overlap):
            x_coords = np.where(np.any(overlap, axis=0))[0]
            mid_x = (x_coords[0] + x_coords[-1]) // 2
            mask[:, :mid_x] = 1.0
            # 过渡带（10像素）
            transition = 10
            if mid_x + transition < new_w:
                mask[:, mid_x:mid_x + transition] = np.linspace(1, 0, transition, endpoint=False)
        else:
            mask[:, :new_w // 2] = 1.0

        # 金字塔融合
        gp_l = [img_l.astype(np.float32)]
        gp_r = [img_r.astype(np.float32)]
        gp_m = [mask]

        for i in range(levels):
            gp_l.append(cv2.pyrDown(gp_l[-1]))
            gp_r.append(cv2.pyrDown(gp_r[-1]))
            gp_m.append(cv2.pyrDown(gp_m[-1]))

        lp_l = [gp_l[-1]]
        lp_r = [gp_r[-1]]

        for i in range(levels, 0, -1):
            size = (gp_l[i - 1].shape[1], gp_l[i - 1].shape[0])
            L = cv2.subtract(gp_l[i - 1], cv2.pyrUp(gp_l[i], dstsize=size))
            R = cv2.subtract(gp_r[i - 1], cv2.pyrUp(gp_r[i], dstsize=size))
            lp_l.append(L)
            lp_r.append(R)

        LS = []
        for l, r, m in zip(lp_l, lp_r, gp_m[::-1]):
            m3 = np.stack([m] * 3, axis=2)
            LS.append(l * m3 + r * (1 - m3))

        res = LS[0]
        for i in range(1, levels + 1):
            size = (LS[i].shape[1], LS[i].shape[0])
            res = cv2.pyrUp(res, dstsize=size)
            res = cv2.add(res, LS[i])

        return np.clip(res[:h, :w], 0, 255).astype(np.uint8)

'''

        # C. 生成权重图 (Feathering / Alpha mask)
        # 我们希望图像中心的像素权重为 1，边缘为 0
        def get_alpha_mask(h, w):
            # 创建线性渐变权重：边缘向中心增加
            mask = np.zeros((h, w), dtype=np.float32)
            # 简单的水平权重：左边缘0 -> 中间1 <- 右边缘0
            for i in range(w):
                mask[:, i] = 1.0 - abs(i - w / 2) / (w / 2)
            # 也可以使用更平滑的平顶函数或高斯分布
            return mask

        mask1 = get_alpha_mask(h1, w1)
        mask2 = get_alpha_mask(h2, w2)

        # 将权重图也进行变换，对齐到画布
        warped_mask1 = np.zeros((output_h, output_w), dtype=np.float32)
        warped_mask1[roi_y:roi_y + h1, roi_x:roi_x + w1] = mask1

        warped_mask2 = cv2.warpPerspective(mask2, full_transform, (output_w, output_h))

        # D. 执行加权融合
        # 避免除以 0
        sum_mask = warped_mask1 + warped_mask2
        sum_mask[sum_mask == 0] = 1.0

        # 归一化权重
        alpha1 = warped_mask1 / sum_mask
        alpha2 = warped_mask2 / sum_mask

        # 融合三通道
        final_img = np.zeros_like(warped_left)
        for c in range(3):
            final_img[:, :, c] = (warped_left[:, :, c] * alpha1 +
                                  warped_right[:, :, c] * alpha2)

        return final_img.astype(np.uint8)
'''

import cv2
import numpy as np


class SimpleStitcher:
    def __init__(self, method='laplacian'):
        self.method = method  # 'average', 'laplacian', 'poisson'

    def stitch(self, img_left, img_right, direction=1):
        # --- [步骤 1-5 保持不变：特征匹配与矩阵计算] ---
        # ... (此处省略你原有的灰度转化、SIFT、筛选、M 矩阵计算代码) ...
        # 1. 转灰度
        g1 = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # 2. 特征提取 (根据OpenCV版本调整，新版无需xfeatures2d)
        try:
            sift = cv2.SIFT_create()
        except AttributeError:
            sift = cv2.xfeatures2d.SIFT_create()

        kp1, des1 = sift.detectAndCompute(g1, None)
        kp2, des2 = sift.detectAndCompute(g2, None)

        # 3. 匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []

        # 4. 筛选与约束 (导师重点：垂直拉伸规律一致 -> y坐标偏差极小)
        # 阈值设定：允许大约 image height 1% - 2% 的垂直误差，视具体图片质量而定
        y_tolerance = img_left.shape[0] * 0.02

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                # 获取特征点坐标
                pt1 = kp1[m.queryIdx].pt  # img_left 的点
                pt2 = kp2[m.trainIdx].pt  # img_right 的点

                # [关键约束]：由于是圆柱面展开后的平移拼接，
                # 同一个特征点在左右两图中的 Y 坐标应该几乎一致（或有固定的微小整体偏移）
                if abs(pt1[1] - pt2[1]) < y_tolerance:
                    good.append(m)

        if len(good) < 10:
            print("匹配点不足")
            return None
        # 6. 计算画布大小与变换
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]

        '''
        # 提取坐标点
        # 注意：我们要把 img_right (dst) 变换对齐到 img_left (src)
        # 所以计算矩阵时，src 是 right_pts, dst 是 left_pts
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

        # 5. 计算变换矩阵

        # M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        base_img = img_left
        warp_img = img_right
        base_h, base_w = h1, w1
        warp_h, warp_w = h2, w2

        '''

        if direction == 1:
            # 以左图为基准：将右图变换到左图坐标系
            src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            base_img = img_left
            warp_img = img_right
            base_h, base_w = h1, w1
            warp_h, warp_w = h2, w2
        else:
            # 以右图为基准：将左图变换到右图坐标系（矩阵求逆）
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            base_img = img_right
            warp_img = img_left
            base_h, base_w = h2, w2
            warp_h, warp_w = h1, w1

        if M is None:
            print("无法计算变换矩阵")
            return None

        # 6. 计算画布大小与变换
        corners_base = np.float32([[0, 0], [0, base_h], [base_w, base_h], [base_w, 0]]).reshape(-1, 1, 2)
        corners_warp = np.float32([[0, 0], [0, warp_h], [warp_w, warp_h], [warp_w, 0]]).reshape(-1, 1, 2)

        # transformed_corners = cv2.transform(corners_right, M)
        transformed_corners = cv2.perspectiveTransform(corners_warp, M)
        all_points = np.concatenate((corners_base, transformed_corners), axis=0)

        [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

        # M_extended = np.vstack([M, [0, 0, 1]])
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
        M_extended = M
        full_transform = H_translation.dot(M_extended)

        output_w, output_h = x_max - x_min, y_max - y_min

        # 7. 创建带权重的融合
        # 将变换图投影到画布
        warped_img = cv2.warpPerspective(warp_img, full_transform, (output_w, output_h))

        # 将基准图放入画布
        warped_base = np.zeros_like(warped_img)
        roi_x, roi_y = translation_dist[0], translation_dist[1]
        warped_base[roi_y:roi_y + base_h, roi_x:roi_x + base_w] = base_img

        # 根据方向调整融合时的参数传递（保持左图在左、右图在右的视觉习惯）
        if direction == 1:
            # 以左图为基准：warped_base是左图，warped_img是右图
            img_l, img_r = warped_base, warped_img
            raw_r = img_right
        else:
            # 以右图为基准：warped_base是右图，warped_img是左图
            # 但为了融合算法的一致性（假设左图在左），这里交换一下
            img_l, img_r = warped_img, warped_base
            raw_r = img_left

        if self.method == 'average':
            return self._blend_average(img_l, img_r)
        elif self.method == 'laplacian':
            return self._blend_laplacian(img_l, img_r)
        elif self.method == 'poisson':
            return self._blend_poisson(img_l, img_r, raw_r, full_transform)

    def _blend_average(self, img_l, img_r):
        mask_l = (np.sum(img_l, axis=2) > 0).astype(np.float32)
        mask_r = (np.sum(img_r, axis=2) > 0).astype(np.float32)

        # 使用距离变换生成羽化权重
        dist_l = cv2.distanceTransform(mask_l.astype(np.uint8), cv2.DIST_L2, 3)
        dist_r = cv2.distanceTransform(mask_r.astype(np.uint8), cv2.DIST_L2, 3)

        sum_dist = dist_l + dist_r
        sum_dist[sum_dist == 0] = 1.0

        alpha = dist_l / sum_dist
        res = np.zeros_like(img_l)
        for c in range(3):
            res[:, :, c] = img_l[:, :, c] * alpha + img_r[:, :, c] * (1 - alpha)
        return res.astype(np.uint8)

    # 2. 拉普拉斯金字塔融合 (解决鬼影与光照不均的最佳平衡点)
    def _blend_laplacian(self, img_l, img_r, levels=5):
        # 强制图像大小为 2^n 对齐
        h, w = img_l.shape[:2]
        new_h, new_w = int(np.ceil(h / 2 ** levels) * 2 ** levels), int(np.ceil(w / 2 ** levels) * 2 ** levels)
        img_l = cv2.copyMakeBorder(img_l, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT)
        img_r = cv2.copyMakeBorder(img_r, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT)

        # 创建重叠区域的掩码 (左图为1, 右图为0, 交界处0.5)
        mask = np.zeros((new_h, new_w), dtype=np.float32)
        mask_l = (np.sum(img_l, axis=2) > 0)
        mask_r = (np.sum(img_r, axis=2) > 0)
        overlap = (mask_l & mask_r)

        mask[mask_l] = 1.0
        if overlap.any():
            # 在重叠中心划线作为融合缝
            overlap_indices = np.where(overlap)
            mid_x = int(np.median(overlap_indices[1]))
            mask[:, mid_x:] = 0.0

        # 构建高斯金字塔
        gp_l, gp_r, gp_m = [img_l.astype(np.float32)], [img_r.astype(np.float32)], [mask]
        for i in range(levels):
            gp_l.append(cv2.pyrDown(gp_l[-1]))
            gp_r.append(cv2.pyrDown(gp_r[-1]))
            gp_m.append(cv2.pyrDown(gp_m[-1]))

        # 构建拉普拉斯金字塔
        lp_l, lp_r = [gp_l[levels]], [gp_r[levels]]
        for i in range(levels, 0, -1):
            L = cv2.subtract(gp_l[i - 1], cv2.pyrUp(gp_l[i], dstsize=(gp_l[i - 1].shape[1], gp_l[i - 1].shape[0])))
            R = cv2.subtract(gp_r[i - 1], cv2.pyrUp(gp_r[i], dstsize=(gp_r[i - 1].shape[1], gp_r[i - 1].shape[0])))
            lp_l.append(L);
            lp_r.append(R)

        # 融合
        LS = []
        for l, r, m in zip(lp_l, lp_r, gp_m[::-1]):
            m3 = cv2.merge([m, m, m])
            LS.append(l * m3 + r * (1.0 - m3))

        # 重建
        res = LS[0]
        for i in range(1, levels + 1):
            res = cv2.pyrUp(res, dstsize=(LS[i].shape[1], LS[i].shape[0]))
            res = cv2.add(res, LS[i])

        return np.clip(res[:h, :w], 0, 255).astype(np.uint8)

    # 3. 泊松融合 (解决色差，但对重影敏感)
    def _blend_poisson(self, img_l, img_r, raw_r, full_M):
        # 1. 找到右图在画布上的有效掩码
        mask = np.zeros(img_r.shape[:2], dtype=np.uint8)
        raw_h, raw_w = raw_r.shape[:2]
        raw_mask = np.ones((raw_h, raw_w), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(raw_mask, full_M, (img_r.shape[1], img_r.shape[0]))

        # 2. 计算中心点
        y, x = np.where(warped_mask > 0)
        center = (int((np.min(x) + np.max(x)) / 2), int((np.min(y) + np.max(y)) / 2))

        # 3. 泊松无缝融合 (将右图缝合到左图底色上)
        # 注意：泊松融合不支持全图大尺寸，如果拼接图过大，可能会报错或极慢
        try:
            return cv2.seamlessClone(img_r, img_l, warped_mask, center, cv2.NORMAL_CLONE)
        except:
            print("泊松融合失败，返回平均融合结果")
            return self._blend_average(img_l, img_r)