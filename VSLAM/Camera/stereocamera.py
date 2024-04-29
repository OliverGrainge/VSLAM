from typing import List

import cv2
import numpy as np

from ..utils import unhomogenize
from .base import ABCCamera
from ..Features import LocalFeatures
from ..utils import pts2kp
from ..FeatureMatchers import FlannMatcher

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
        lowe_ratio: float=0.8,
        reproj_error_threshold: float=8.0
    ):
        
        self.feature_extractor = LocalFeatures()
        self.feature_matcher = FlannMatcher()
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
        self = self.feature_matcher.match(self)
        #self.features_matcher.match(self)
        #self.get_matches()
        #self.filter_matching_inliers()
        #reproj_error = self.triangulate()
        #self.filter_triangulated_points(reproj_error)


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

    def get_matches(self):
        detector = cv2.SIFT_create(nfeatures=500)

        keyPointsLeft, descriptorsLeft = detector.detectAndCompute(self.left_image, None)
        keyPointsRight, descriptorsRight = detector.detectAndCompute(self.right_image, None)

        matcher = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))

        matches = matcher.knnMatch(descriptorsLeft, descriptorsRight, 2)

        # apply ratio test
        queryidxs = [m.queryIdx for m, n in matches if m.distance < 0.8 * n.distance]
        trainidxs = [m.trainIdx for m, n in matches if m.distance < 0.8 * n.distance]

        ptsLeft = np.array(cv2.KeyPoint_convert(keyPointsLeft))[queryidxs]
        ptsRight = np.array(cv2.KeyPoint_convert(keyPointsRight))[trainidxs]

        ptsLeft = np.array(ptsLeft).astype('float64')
        ptsRight = np.array(ptsRight).astype('float64')

        self.left_kp = pts2kp(ptsLeft)
        self.right_kp = pts2kp(ptsRight)
        self.left_desc2d = descriptorsLeft[queryidxs]
        self.right_desc2d = descriptorsRight[trainidxs]
        self.left_kpoints2d = np.array(ptsLeft).squeeze()
        self.right_kpoints2d = np.array(ptsRight).squeeze()


    def filter_matching_inliers(self,):
        _, mask = cv2.findEssentialMat(self.left_kpoints2d,
                                        self.right_kpoints2d,
                                        self.kl,
                                        method = 8,
                                        prob = 0.9999,
                                        threshold = 0.8)
        mask = mask.ravel().astype(bool)
        self.left_kp = self.left_kp[mask]
        self.right_kp = self.right_kp[mask]
        self.left_desc2d = self.left_desc2d[mask]
        self.right_desc2d = self.right_desc2d[mask]
        self.left_kpoints2d = self.left_kpoints2d[mask]
        self.right_kpoints2d = self.right_kpoints2d[mask]
        
    def triangulate(self):
        pts4D = cv2.triangulatePoints(self.pl, self.pr, self.left_kpoints2d.T, self.right_kpoints2d.T)
        pts3D = pts4D[:3,:]/((pts4D[-1,:]).reshape(1,-1))
        pts3D = pts3D.T
        proj2D_left = project_points(pts3D, self.pl)
        proj2D_right = project_points(pts3D, self.pr)
        reprojError = ((np.sqrt(((proj2D_left-self.left_kpoints2d)**2).sum(axis=1))) + (np.sqrt(((proj2D_right-self.right_kpoints2d)**2).sum(axis=1))))/2
        self.kpoints3d = np.array(pts3D)
        return reprojError


    def filter_triangulated_points(self, reprojError):
        mask_x = np.logical_and((self.kpoints3d[:, 0] > - 12), (self.kpoints3d[:, 0] < 12))
        mask_y = np.logical_and((self.kpoints3d[:, 1] < 2), (self.kpoints3d[:, 1] > -8))
        mask_z = (self.kpoints3d[:, 2] > 2)
        mask_reproj = reprojError < 0.5
        
        mask = np.logical_and(np.logical_and(mask_x, mask_y, mask_z), mask_reproj)
        self.kpoints3d = self.kpoints3d[mask]
        self.left_kp = self.left_kp[mask]
        self.right_kp = self.right_kp[mask]
        self.left_desc2d = self.left_desc2d[mask]
        self.right_desc2d = self.right_desc2d[mask]
        self.left_kpoints2d = self.left_kpoints2d[mask]
        self.right_kpoints2d = self.right_kpoints2d[mask]

