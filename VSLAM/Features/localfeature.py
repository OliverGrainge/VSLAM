from typing import List, Tuple, Union

import cv2
import numpy as np

import VSLAM.Features.Local.Describers as Describers
import VSLAM.Features.Local.Detectors as Detectors

from ..utils import get_config

config = get_config()


def get_detector():
    if config["LocalFeatureDetector"] == "SIFT":
        return Detectors.SIFT()
    elif config["LocalFeatureDetector"] == "HARRIS":
        return Detectors.HARRIS()
    elif config["LocalFeatureDetector"] == "SIFTBlocks": 
        return Detectors.SIFTBlocks()
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
        self.type = f"detector {config['LocalFeatureDetector']} describer {config['LocalFeatureDescriber']}"

    def compute(self, camera):
        camera.left_desc2d = self.describer.compute(camera.left_image, camera.left_kp)

        camera.right_desc2d = self.describer.compute(
            camera.right_image, camera.right_kp
        )

        return camera

    def detect(self, camera):
        camera.left_kp = np.array(self.detector.detect(camera.left_image))
        camera.left_kpoints2d = cv2.KeyPoint_convert(camera.left_kp)
        camera.right_kp = np.array(self.detector.detect(camera.right_image))
        camera.right_kpoints2d = cv2.KeyPoint_convert(camera.right_kp)
        return camera

    def detectAndCompute(self, camera):
        camera = self.detect(camera)
        camera = self.compute(camera)
        return camera
