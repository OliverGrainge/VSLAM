import cv2
import numpy as np
from typing import Tuple, List
from ...utils import get_config
from .base import ABCLocalFeature

config = get_config()


class ShiTomasiSIFT(ABCLocalFeature):
    def __init__(self):
        # Create SIFT object for descriptor computation
        self.sift = cv2.SIFT_create(nfeatures=config["Stereo"]["MaxFeatures"])
        # Parameters for Shi-Tomasi corner detection
        self.maxCorners = config["Stereo"]["MaxFeatures"]
        self.qualityLevel = 0.01
        self.minDistance = 10
        self.blockSize = 3
        self.useHarrisDetector = False
        self.k = 0.04

    def detect(self, image: np.ndarray) -> np.ndarray:
        corners = cv2.goodFeaturesToTrack(
            image,
            self.maxCorners,
            self.qualityLevel,
            self.minDistance,
            blockSize=self.blockSize,
            useHarrisDetector=self.useHarrisDetector,
            k=self.k,
        )
        if corners is not None:
            keypoints = [
                cv2.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=10)
                for pt in corners
            ]
        else:
            keypoints = []
        return np.array(keypoints)

    def compute(self, image: np.ndarray, keypoints: List[cv2.KeyPoint]) -> np.ndarray:
        _, desc = self.sift.compute(image, keypoints)
        return desc

    def detectAndCompute(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        keypoints = self.detect(image)
        desc = self.compute(image, keypoints)
        return keypoints, desc
