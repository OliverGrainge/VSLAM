import numpy as np
import cv2
from ..utils import homogenize


class MotionEstimation3D2D:
    def __init__(
        self,
        reprojErrorThreshold: float = 8.0,
    ):
        self.reprojErrorThreshold = reprojErrorThreshold

    def __call__(self, camera1, camera2):
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            camera1.kpoints3d,
            camera2.left_kpoints2d,
            camera2.kl,
            camera2.dist,
            reprojectionError=self.reprojErrorThreshold,
        )
        """
        kpoints3d_inliers = camera1.kpoints3d[inliers.flatten()]
        left_points2d_inliers = camera1.left_kpoints2d[inliers.flatten()]
        
        rvec, tvec = cv2.solvePnPRefineLM(
            kpoints3d_inliers, 
            left_points2d_inliers, 
            camera2.kl,
            camera2.dist, 
            rvec, 
            tvec
        )
        """

        T = np.linalg.inv(homogenize(rvec, tvec))
        return T
