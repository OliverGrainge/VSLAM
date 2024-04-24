from .base import ABCDetector
import cv2
from typing import List 
import numpy as np

class SIFT(ABCDetector):
    def __init__(self): 
        self.sift = cv2.SIFT_create()

    def detect(self, image: np.ndarray) -> List:
        keypoints = self.sift.detect(image, None)
        return keypoints
