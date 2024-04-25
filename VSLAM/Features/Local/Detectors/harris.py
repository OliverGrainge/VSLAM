from typing import List

import cv2
import numpy as np

from VSLAM.utils import pts2kp

from .base import ABCDetector


class HARRIS(ABCDetector):
    def detect(self, image: np.ndarray) -> List:
        pts = cv2.goodFeaturesToTrack(image, 100, 0.01, 10)
        keypoints = pts2kp(pts)
        return keypoints
