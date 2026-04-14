import cv2
import numpy as np

class DistortionCorrector:
    def __init__(self, params_path='camera_params.npz'):
        data = np.load(params_path)
        self.mtx = data['mtx']
        self.dist = data['dist']
        # 提取标定得到的实际焦距（取 fx 和 fy 的平均值）
        self.focal_length = (self.mtx[0, 0] + self.mtx[1, 1]) / 2.0

    def process(self, img):
        h, w = img.shape[:2]
        # alpha=0 表示剪掉黑边，保持画面干净
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 0, (w, h))
        dst = cv2.undistort(img, self.mtx, self.dist, None, new_mtx)
        return dst