from .base import ABCCamera
import numpy as np
from utils import unhomogenize
import cv2
from typing import List

class StereoCamera(ABCCamera):
    def __init__(
        self,
        x: np.ndarray=None,
        k: np.ndarray=None,
        pl: np.ndarray=None,
        pr: np.ndarray=None,
        dist: np.ndarray=np.zeros(4),
    ):

        self.x = x              # extrinsic parameters of left camera
        self.k = k              # intrinsic parameters of left cmaera
        self.pl = pl            # left projection matrix
        self.pr = pr            # right projection matrix 
        self.dist = dist        # distortion parameters

        self.kp = []            # List of cv2 keypoints for left image
        self.kpoints2d = None   # Locations of keypoints for left image
        self.kpoints3d = None   # Locations of 3d Keypoints
        self.desc2d = None      # descriptors of 2d keypoints
        self.desc3d = None      # descriptions of 3d keypoints

    
    def project(self, points: np.ndarray):
        rvec, tvec = unhomogenize(self._x)
        projected_points, _ = cv2.projectPoints(
            points.T,
            rvec, 
            tvec, 
            self.k, 
            self.dist)
        return projected_points

    