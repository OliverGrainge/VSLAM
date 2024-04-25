from typing import List

import cv2
import numpy as np

from ..utils import unhomogenize
from .base import ABCCamera


class StereoCamera(ABCCamera):
    def __init__(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        pl: np.ndarray,
        pr: np.ndarray,
        kl: np.ndarray,
        kr: np.ndarray,
        x: np.ndarray = np.eye(4),
        dist: np.ndarray = np.zeros(5),
    ):
        self.left_image = left_image
        self.right_image = right_image

        self.x = x  # extrinsic parameters of left camera
        self.kl = kl  # intrinsic parameters of left camera
        self.kr = kr  # intrinsic parameters of right camera
        self.pl = pl  # left projection matrix
        self.pr = pr  # right projection matrix
        self.dist = dist  # distortion parameters

        self.left_kp = []  # List of cv2 keypoints for left image
        self.right_kp = []  # List of cv2 keypoints for right image
        self.left_kpoints2d = None  # Locations of keypoints for left image
        self.right_kpoints2d = None  # Locations of keypoints for right image
        self.kpoints3d = None  # Locations of 3d Keypoints
        self.left_desc2d = None  # descriptors of 2d keypoints
        self.right_desc2d = None  # descriptors of 2d keypoints
        self.desc3d = None  # descriptions of 3d keypoints

    def project(self, points: np.ndarray):
        rvec, tvec = unhomogenize(self.x)
        projected_points, _ = cv2.projectPoints(points.T, rvec, tvec, self.k, self.dist)
        return projected_points
