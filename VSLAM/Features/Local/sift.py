import cv2
import numpy as np
from typing import Tuple, List
from ...utils import get_config
from .base import ABCLocalFeature

config = get_config()


class SIFT(ABCLocalFeature):
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=config["Stereo"]["MaxFeatures"])

    def detect(self, image: np.ndarray) -> np.ndarray:
        keypoints = self.sift.detect(image, None)
        return np.array(keypoints)

    def compute(self, image: np.ndarray, keypoints: List) -> np.ndarray:
        desc = self.sift.compute(image, keypoints)[1]
        return desc

    def detectAndCompute(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        keypoints, desc = self.sift.detectAndCompute(image, None)
        return np.array(keypoints), desc
