from .base import ABCDescriber
from typing import List, Union
import numpy as np
import cv2

class SIFT(ABCDescriber):
    def __init__(self): 
        self.sift = cv2.SIFT_create()

    def compute(self, image: np.ndarray, keypoints: Union[np.ndarray, List]) -> np.array:
        if isinstance(keypoints, List):
            keypoints = cv2.KeyPoints_convert(keypoints)
        desc = self.sift.compute(image, keypoints)[1]
        return desc
