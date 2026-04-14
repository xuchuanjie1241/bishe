
import cv2
import os
import numpy as np
from config import Config
from modules.locator import CylinderLocator
from modules.preprocessor import DistortionCorrector
from modules.unwarper import CylinderUnwarper
from modules.stitcher import SimpleStitcher


def process_single_image(img_path, corrector, locator, unwarper, focal_length):
    """处理单张图片的流程"""
    # 读取并去畸变
    img_raw = cv2.imread(img_path)
    if img_raw is None:
        print(f"无法读取图片: {img_path}")
        return None

    output_dir = os.path.join(Config.RESULT_DIR, "debug")
    img_undistorted = corrector.process(img_raw)
    img_undistorted = img_raw if Config.DEBUG else img_undistorted
    if not Config.DEBUG:
        cv2.imshow("Undistorted", img_undistorted)
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f"undistorted_{os.path.basename(img_path)}"), img_undistorted)
        cv2.waitKey(0)

    # 定位圆柱体（自动进行半径归一化）
    rect, r, center_info = locator.process(img_undistorted, normalize=True)
    if not Config.DEBUG:
        cv2.imshow("Rectified", rect)
        cv2.imwrite(os.path.join(output_dir, f"rectified_{os.path.basename(img_path)}"), rect)
        cv2.waitKey(0)

    # 展开
    # 根据裁剪后的宽度比例调整焦距
    # scale = rect.shape[1] / img_undistorted.shape[1]
    # f_adjusted = focal_length * scale
    cy = center_info['y']
    flat = unwarper.unwarp(rect, r, focal_length, center_y=cy)

    if not Config.DEBUG:
        print(f"  展开图尺寸: {flat.shape[1]}x{flat.shape[0]}")
        cv2.imshow("Unwarped", flat)
        cv2.imwrite(os.path.join(output_dir, f"unwarped_{os.path.basename(img_path)}"), flat)
        cv2.waitKey(0)

    return flat


def stitch_from_center(flats, stitcher):
    """
    从中间向左右两侧拼接
    flats: 展开后的图像列表（已按环绕顺序排列）
    """
    n = len(flats)
    if n == 0:
        return None
    if n == 1:
        return flats[0]

    # 找到中心索引
    center_idx = n // 2

    # 初始化全景图为中心图像
    panorama = flats[center_idx]

    # 向左拼接（索引减小方向）
    left_idx = center_idx - 1
    while left_idx >= 0:
        print(f"向左拼接: {left_idx} <- {left_idx + 1}")
        # 注意：向左拼接时，新图在左，当前全景图在右
        # stitch(left, right) -> 左图右移，右图作为基准
        panorama = stitcher.stitch(flats[left_idx], panorama, direction=-1)
        if Config.DEBUG:
            cv2.imshow("Stitching", panorama)
            cv2.waitKey(0)
        if panorama is None:
            print(f"向左拼接失败 at {left_idx}")
            break
        left_idx -= 1

    # 向右拼接（索引增加方向）
    # 注意：此时全景图包含了左侧所有图，继续向右拼接
    right_idx = center_idx + 1
    while right_idx < n:
        print(f"向右拼接: {right_idx - 1} -> {right_idx}")
        # 向右拼接时，当前全景图在左，新图在右
        panorama = stitcher.stitch(panorama, flats[right_idx], direction=1)
        if Config.DEBUG:
            cv2.imshow("Stitching", panorama)
            cv2.waitKey(0)
        if panorama is None:
            print(f"向右拼接失败 at {right_idx}")
            break
        right_idx += 1

    return panorama


def main():
    # 初始化模块
    corrector = DistortionCorrector('modules/camera_params.npz')
    locator = CylinderLocator()
    unwarper = CylinderUnwarper()
    stitcher = SimpleStitcher(method='laplacian')

    # 获取图片列表并排序（确保环绕顺序正确）
    image_dir = Config.IMAGE_DIR
    # image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

    image_files = Config.IMAGES
    if len(image_files) < 2:
        print("至少需要2张图片")
        return

    print(f"找到 {len(image_files)} 张图片: {image_files}")

    # 使用第一张图作为基准，设置归一化半径
    print("处理基准图...")
    img0 = cv2.imread(os.path.join(image_dir, image_files[0]))
    img0_undistorted = corrector.process(img0)
    img0_undistorted = img0 if Config.DEBUG else img0_undistorted

    _, r0, _ = locator.process(img0_undistorted, normalize=False)  # 第一张不缩放，设为基准
    locator.set_reference(r0)  # 设置参考半径
    print(f"基准半径: {r0:.1f}px")

    focal_length = corrector.focal_length * 1.1

    # 重新处理第一张图（进行归一化）
    flats = []
    flat0 = process_single_image(
        os.path.join(image_dir, image_files[0]),
        corrector, locator, unwarper,
        focal_length
    )
    if flat0 is not None:
        flats.append(flat0)

    # 处理剩余图片（自动归一化）
    for img_name in image_files[1:]:
        print(f"处理: {img_name}")
        flat = process_single_image(
            os.path.join(image_dir, img_name),
            corrector, locator, unwarper,
            focal_length
        )
        if flat is not None:
            flats.append(flat)

    if len(flats) < 2:
        print("有效图片不足")
        return

    print(f"成功处理 {len(flats)} 张图片，开始拼接...")

    # 从中间向左右拼接
    result = stitch_from_center(flats, stitcher)

    if result is not None:
        output_path = os.path.join(Config.RESULT_DIR, "panorama.jpg")
        cv2.imwrite(output_path, result)
        print(f"结果已保存: {output_path}")

        # 显示结果（按尺寸缩放以适应屏幕）
        display_h = 800
        scale = display_h / result.shape[0]
        display_w = int(result.shape[1] * scale)
        display_img = cv2.resize(result, (display_w, display_h))
        cv2.imshow("Panorama7", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

'''
import cv2
from config import Config
from modules.locator import CylinderLocator
from modules.preprocessor import DistortionCorrector
from modules.unwarper import CylinderUnwarper
from modules.stitcher import SimpleStitcher


def main():
    # 初始化模块
    corrector = DistortionCorrector('modules/camera_params.npz')
    locator = CylinderLocator()
    unwarper = CylinderUnwarper()
    stitcher = SimpleStitcher()
    #
    images = [cv2.imread(path) for path in Config.IMAGE_LIST]
    flats = []

    # 1. 批量预处理与展开
    for img in images:
        rect, r = locator.process(img)
        flat = unwarper.unwarp(rect, r, Config.F)
        flats.append(flat)

    # 2. 级联拼接 (Cascading Stitch)
    panorama = flats[0]
    for i in range(1, len(flats)):
        panorama = stitcher.stitch(panorama, flats[i])
    #
    # 读取图片
    img1_raw_distorted = cv2.imread(Config.IMAGE_DIR + Config.IMAGE_1)
    img2_raw_distorted = cv2.imread(Config.IMAGE_DIR + Config.IMAGE_2)

    # --- 流程开始 ---
    img1_raw = corrector.process(img1_raw_distorted)
    img2_raw = corrector.process(img2_raw_distorted)

    if Config.DEBUG:
        cv2.imshow("Distorted 1", img1_raw)
        cv2.imshow("Distorted 2", img2_raw)
        cv2.imwrite(Config.RESULT_DIR + "distorted" + Config.IMAGE_1, img1_raw)
        cv2.imwrite(Config.RESULT_DIR + "distorted" + Config.IMAGE_2, img2_raw)
        cv2.waitKey(0)

    # 步骤 2-3: 寻找主体 & 位置校正
    img1_rect, r1 = locator.process(img1_raw)
    img2_rect, r2 = locator.process(img2_raw)

    if Config.DEBUG:
        cv2.imshow("Rectified 1", img1_rect)
        cv2.imshow("Rectified 2", img2_rect)
        cv2.imwrite(Config.RESULT_DIR + "rect" + Config.IMAGE_1, img1_rect)
        cv2.imwrite(Config.RESULT_DIR + "rect" + Config.IMAGE_2, img2_rect)
        cv2.waitKey(0)


    # 步骤 4: 几何变换 (展开)
    # 估算焦距：简单的线性关系
    f1 = img1_raw.shape[1] * Config.F_RATIO
    f2 = img2_raw.shape[1] * Config.F_RATIO
    f_calib = corrector.focal_length

    # 1. 获取原图宽度和裁剪后的宽度
    raw_w = img1_raw_distorted.shape[1]  # 原图宽度 (如 1920)
    rect_w = img1_rect.shape[1]  # 裁剪后的瓶身宽度 (如 400)

    # 2. 计算缩放比例
    scale = rect_w / raw_w

    # 3. 按比例调整标定焦距
    # 假设 f_calib 是你标定得到的 1275.3
    f_final = f_calib * scale

    # 4. 传入 unwarper
    # 这样 theta = (x - center_x) / f_final 就能产生足够大的角度变化了

    flat1 = unwarper.unwarp(img1_rect, r1, f_calib)
    flat2 = unwarper.unwarp(img2_rect, r2, f_calib)

    if Config.DEBUG:
        cv2.imshow("Unwarped 1", flat1)
        cv2.imshow("Unwarped 2", flat2)
        cv2.imwrite(Config.RESULT_DIR + "unwarped" + Config.IMAGE_1, flat1)
        cv2.imwrite(Config.RESULT_DIR + "unwarped" + Config.IMAGE_2, flat2)
        cv2.waitKey(0)


    # 步骤 5-6: 特征提取与融合
    # 注意：这里假设 img1 是左边的图，img2 是右边的图
    result = stitcher.stitch(flat1, flat2)

    if result is not None:
        cv2.imwrite(Config.RESULT_DIR + "result" + Config.IMAGE_1, result)
        cv2.imshow("Result", result)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
'''