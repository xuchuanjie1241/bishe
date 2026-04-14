class Config:
    # 调试模式：设为True会显示每一步的中间结果（如边缘检测图、旋转后的图）
    DEBUG = True

    # 图像缩放（处理高分辨率图片时加速）
    PROCESS_WIDTH = 800

    # 几何参数估计
    # 焦距估算系数：假设焦距 f = 图像宽度 * F_RATIO (手机拍摄通常在 1.0 - 2.0 之间)
    F_RATIO = 1

    IMAGE_DIR = "data/raw/"
    RESULT_DIR = "data/output/"

    IMAGE_NS = ["n0001.jpg", "n0002.jpg", "n0003.jpg"]
    IMAGE_ZS = ["z01.jpg", "z02.jpg", "z03.jpg"]
    IMAGE_FS = ["f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg", "f5.jpg"]
    IMAGE_DS = ["d01.jpg", "d02.jpg", "d03.jpg", "d04.jpg", "d05.jpg"]
    IMAGE_CS = ["c1.jpg", "c2.jpg", "c3.jpg", "c4.jpg", "c5.jpg"]


    IMAGE_CHESSBOARD = "cyl_checker_02.jpg"

    IMAGES = IMAGE_DS