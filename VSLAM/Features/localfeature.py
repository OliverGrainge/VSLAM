from typing import List, Tuple, Union

import cv2
import numpy as np

from VSLAM.Features.Local import SIFT, ShiTomasiSIFT

from ..utils import get_config

config = get_config()


def get_feature_extractor():
    if config["LocalFeatureDetector"].lower() == "sift":
        return SIFT()
    elif config["LocalFeatureDetector"].lower() == "shitomasisift":
        return ShiTomasiSIFT()
    else:
        raise NotImplementedError()


class LocalFeatures:
    def __init__(self):
        self.feature_extractor = get_feature_extractor()

    def compute(self, camera):
        camera.left_desc2d = self.feature_extractor.compute(
            camera.left_image, camera.left_kp
        )

        camera.right_desc2d = self.feature_extractor.compute(
            camera.right_image, camera.right_kp
        )

        return camera

    def detect(self, camera):
        camera.left_kp = np.array(self.feature_extractor.detect(camera.left_image))
        camera.left_kpoints2d = cv2.KeyPoint_convert(camera.left_kp)
        camera.right_kp = np.array(self.feature_extractor.detect(camera.right_image))
        camera.right_kpoints2d = cv2.KeyPoint_convert(camera.right_kp)
        return camera

    def detectAndCompute(self, camera):
        camera.left_kp, camera.left_desc2d = self.feature_extractor.detectAndCompute(
            camera.left_image
        )
        camera.right_kp, camera.right_desc2d = self.feature_extractor.detectAndCompute(
            camera.right_image
        )
        camera.left_kpoints2d = np.array(cv2.KeyPoint_convert(camera.left_kp))
        camera.right_kpoints2d = np.array(cv2.KeyPoint_convert(camera.right_kp))
        return camera
