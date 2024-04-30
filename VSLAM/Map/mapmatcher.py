import numpy as np
import cv2
from VSLAM.Camera.stereocamera import StereoCamera
from typing import List



class MapMatcher:
    def __init__(self):
        pass

    def match(self, points3D: np.ndarray, desc3d: np.ndarray, cameraidxs: np.ndarray, cameras: List[StereoCamera]):
        """
        This function takes in the 3d points and descriptors of the points. Matches those points
        with the camera observations and returns a list of (i, k, j) where the ith 3d point is
        matched to the jth observation sof the kth camera.

        :param points3D:
        :param desc3d:
        :param camera:
        :return:
        """
        pass
