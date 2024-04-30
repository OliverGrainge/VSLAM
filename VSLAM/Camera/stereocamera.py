from typing import List

import cv2
import numpy as np

from ..utils import unhomogenize
from .base import ABCCamera
from ..Features import LocalFeatures
from ..utils import pts2kp
from ..FeatureMatchers import FeatureMatcher

def project_points(points3D, projectionMatrix):
    """
    :param points3D (numpy.array) : size (Nx3)
    :param projectionMatrix (numpy.array) : size(3x4) - final projection matrix (K@[R|t])
    
    Returns:
        points2D (numpy.array) : size (Nx2) - projection of 3D points on image plane
    """
    
    points3D = np.hstack((points3D,np.ones(points3D.shape[0]).reshape(-1,1))) #shape:(Nx4)
    points3D = points3D.T #shape:(4xN)
    pts2D_homogeneous = projectionMatrix @ points3D #shape:(3xN)
    pts2D = pts2D_homogeneous[:2, :]/(pts2D_homogeneous[-1,:].reshape(1,-1)) #shape:(2xN)
    pts2D = pts2D.T
                         
    return pts2D



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
        
        self.feature_extractor = LocalFeatures()
        self.feature_matcher = FeatureMatcher()
    
        self.left_image = left_image
        self.right_image = right_image

        self.x = x  # extrinsic parameters of left camera
        self.kl = kl  # intrinsic parameters of left camera
        self.kr = kr  # intrinsic parameters of right camera
        self.pl = pl  # left projection matrix
        self.pr = pr  # right projection matrix
        self.dist = dist

        self.left_kp = []  # List of cv2 keypoints for left image
        self.right_kp = []  # List of cv2 keypoints for right image
        self.left_kpoints2d = None  # Locations of keypoints for left image
        self.right_kpoints2d = None  # Locations of keypoints for right image
        self.kpoints3d = None  # Locations of 3d Keypoints
        self.left_desc2d = None  # descriptors of 2d keypoints
        self.right_desc2d = None  # descriptors of 2d keypoints
        self.desc3d = None  # descriptions of 3d keypoints

        self.baseline_vector = self.baseline()
        self.feature_extractor.detectAndCompute(self)
        self.feature_matcher.match(self)



    def baseline(self):
        kl, rl, tl = cv2.decomposeProjectionMatrix(self.pl)[:3]
        kr, rr, tr = cv2.decomposeProjectionMatrix(self.pr)[:3]
        camera_center_left = -np.dot(np.linalg.inv(rl), tl[:3] / tl[3])
        camera_center_right = -np.dot(np.linalg.inv(rr), tr[:3] / tr[3])
        baseline_vector = camera_center_right - camera_center_left
        return baseline_vector


    def project(self, points: np.ndarray):
        rvec, tvec = unhomogenize(self.x)
        projected_points, _ = cv2.projectPoints(
            points.T, rvec, tvec, self.kl, self.dist
        )
        return projected_points.squeeze()

