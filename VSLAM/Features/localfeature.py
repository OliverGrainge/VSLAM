import cv2
import VSLAM.Features.Local.Detectors as Detectors
import VSLAM.Features.Local.Describers as Describers
from ..utils import get_config
import numpy as np
from typing import Tuple, Union, List

config = get_config()

def get_detector(): 
    if config["LocalFeatureDetector"] == "SIFT":
        return Detectors.SIFT()
    else: 
        raise NotImplementedError()


def get_describer(): 
    if config["LocalFeatureDescriber"] == "SIFT":
        return Describers.SIFT()
    else: 
        raise NotImplementedError()



class LocalFeatures:
    def __init__(self):
        self.detector = get_detector()
        self.describer = get_describer()

    def compute(self, image: np.ndarray, keypoints: Union[np.ndarray, List]) -> np.array:
        return self.describer.compute(image, keypoints)

    def detect(self, image: np.ndarray) -> List:
        return self.detector.detect(image)

    def detectAndCompute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        keypoints = self.detect(image)
        desc = self.compute(image, keypoints)
        return keypoints, desc

