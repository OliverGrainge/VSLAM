from typing import List

import cv2
import numpy as np

from ..utils import unhomogenize
from .base import ABCCamera
from ..Features import LocalFeatures


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
        lowe_ratio: float=0.75,
        reproj_error_threshold: float=8.0
    ):
        
        self.feature_extractor = LocalFeatures()
        self.lowe_ratio = lowe_ratio
        self.reproj_error_threshold = reproj_error_threshold
    
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

        self.feature_extractor.detectAndCompute(self)
        self.get_matches()
        self.filter_inliers2d()
        self.triangulate()
        self.filter_inliers3d()
        
    
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

    def get_matches(self):
        if self.left_desc2d.dtype == np.float32:
            matcher = cv2.BFMatcher()
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


        matches = matcher.knnMatch(self.left_desc2d, self.right_desc2d, k=2)
        # apply lowe 
        matches = [m for m, n in matches if m.distance < self.lowe_ratio * n.distance]
        
        new_left_kp, new_right_kp = [], []
        new_left_desc, new_right_desc = [], []
        for match in matches:
            new_left_kp.append(self.left_kp[match.queryIdx])
            new_right_kp.append(self.right_kp[match.trainIdx])
            new_left_desc.append(self.left_desc2d[match.queryIdx])
            new_right_desc.append(self.right_desc2d[match.trainIdx])

        self.left_kp = np.array(new_left_kp).squeeze()
        self.right_kp = np.array(new_right_kp).squeeze()
        self.left_desc2d = np.array(new_left_desc).squeeze()
        self.right_desc2d = np.array(new_right_desc).squeeze()
        self.left_kpoints2d = np.array(cv2.KeyPoint_convert(self.left_kp)).squeeze()
        self.right_kpoints2d = np.array(cv2.KeyPoint_convert(self.right_kp)).squeeze()
        

    def filter_inliers2d(self):
        _, mask = cv2.findHomography(self.left_kpoints2d, self.right_kpoints2d, cv2.RANSAC, self.reproj_error_threshold)
        self.left_kp = self.left_kp[mask.flatten()==1].flatten()
        self.right_kp = self.right_kp[mask.flatten()==1].flatten()
        self.left_desc2d = self.left_desc2d[mask.flatten()==1]
        self.right_desc2d = self.right_desc2d[mask.flatten()==1]
        self.left_kpoints2d = self.left_kpoints2d[mask.flatten()==1].squeeze()
        self.right_kpoints2d = self.right_kpoints2d[mask.flatten()==1].squeeze()
        

    def triangulate(self):
        self.desc3d = self.left_desc2d
        points_4d = cv2.triangulatePoints(
            self.pl, self.pr, self.left_kpoints2d.T, self.right_kpoints2d.T
        )
        points_3d = points_4d[:3] / points_4d[3]
        self.kpoints3d = points_3d.T.squeeze()


    def filter_inliers3d(self):
        rvec, tvec = unhomogenize(self.x)
        proj_left, _ = cv2.projectPoints(
            self.kpoints3d.T, rvec, tvec, self.kl, self.dist
        )
        proj_left = proj_left.squeeze()
        reproj_error = np.linalg.norm(proj_left - self.left_kpoints2d, axis=1)
        mask_dist_min = self.kpoints3d[:, 2] > 2.0
        mask_dist_max = self.kpoints3d[:, 2] < 30
        mask_err = reproj_error < self.reproj_error_threshold
        mask = np.logical_and(mask_dist_min, mask_dist_max, mask_err)
        self.left_kp = self.left_kp[mask]
        self.right_kp = self.right_kp[mask]
        self.left_desc2d = self.left_desc2d[mask]
        self.right_desc2d = self.right_desc2d[mask]
        self.left_kpoints2d = self.left_kpoints2d[mask]
        self.right_kpoints2d = self.right_kpoints2d[mask]