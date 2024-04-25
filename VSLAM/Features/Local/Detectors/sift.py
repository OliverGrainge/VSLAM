from typing import List

import cv2
import numpy as np

from .base import ABCDetector


class SIFT(ABCDetector):
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def detect(self, image: np.ndarray) -> List:
        keypoints = self.sift.detect(image, None)
        return keypoints
