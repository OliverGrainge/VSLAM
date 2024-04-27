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
        self.baseline = self.compute_baseline()

        self.left_kp = []  # List of cv2 keypoints for left image
        self.right_kp = []  # List of cv2 keypoints for right image
        self.left_kpoints2d = None  # Locations of keypoints for left image
        self.right_kpoints2d = None  # Locations of keypoints for right image
        self.kpoints3d = None  # Locations of 3d Keypoints
        self.left_desc2d = None  # descriptors of 2d keypoints
        self.right_desc2d = None  # descriptors of 2d keypoints
        self.desc3d = None  # descriptions of 3d keypoints

        # these masks are to provide mappings between kp and kpoints2d for instance
        # self.left_kp[self.left_mask].pt = left_kpoints2d_kp
        # self.left_desc2d[self.left_mask] = left_kpoint2d_desc
        self.orig_mask = None
        self.targ_mask = None
    
    def compute_baseline(self):
        _,tl = cv2.decomposeProjectionMatrix(self.pl)[1:3]
        _,tr = cv2.decomposeProjectionMatrix(self.pr)[1:3]
        return np.linalg.norm(tl-tr)


    def project(self, points: np.ndarray):
        rvec, tvec = unhomogenize(self.x)
        projected_points, _ = cv2.projectPoints(
            points.T, rvec, tvec, self.kl, self.dist
        )
        return projected_points.squeeze()

    def triangulate(self):
        self.desc3d = self.left_desc2d
        points_4d = cv2.triangulatePoints(
            self.pl, self.pr, self.left_kpoints2d.T, self.right_kpoints2d.T
        )

        if np.allclose(self.x, np.eye(4), atol=1e-6):
            points_3d = points_4d[:3] / points_4d[3]
            depths = points_3d[2, :]
            # filter out points triangulated far away
            mask_depth = depths < 40 * self.baseline
            mask_neg = depths > 0
            mask = mask_depth & mask_neg  
            self.kpoints3d = points_3d.T
            self.kpoints3d = self.kpoints3d[mask]
            self.left_desc2d = self.left_desc2d[mask]
            self.left_kp = self.left_kp[mask]
            self.left_kpoints2d = self.left_kpoints2d[mask]
            self.right_kpoints2d = self.right_kpoints2d[mask]
            if self.right_desc2d is not None:
                self.right_desc2d = self.right_desc2d[mask]

        else:
            # if camera is not at the origin tranlate the points
            points_3d = points_4d[:3] / points_4d[3]
            depths = points_3d[2, :]
            points_4d = self.x @ points_4d
            points_4d /= points_4d[3, :]
            mask_depth = depths < 40 * self.baseline
            mask_neg = depths > 0
            mask = mask_depth & mask_neg 
            self.kpoints3d = points_4d[:3, :].T
            self.left_desc2d = self.left_desc2d[mask]
            self.kpoints3d = self.kpoints3d[mask]
            self.left_kp = self.left_kp[mask]
            self.left_kpoints2d = self.left_kpoints2d[mask]
            self.right_kpoints2d = self.right_kpoints2d[mask]

            if self.right_desc2d is not None:
                self.right_desc2d = self.right_desc2d[mask]
            if self.right_kp is not None: 
                self.right_kp = self.right_kp[mask]

        return self.kpoints3d
