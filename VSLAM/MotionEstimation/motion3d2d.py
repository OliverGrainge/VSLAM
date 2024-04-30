import numpy as np
import cv2
from ..utils import get_config

config = get_config()



class MotionEstimation3D2D:

    def __call__(self, tracking_info: dict):
        _, rvec, tvec, inliers = cv2.solvePnPRansac(tracking_info["kpoints3d_prev"],
                                        tracking_info["kpoints2d_left_cur"],
                                        tracking_info["kl"],
                                        None,
                                        iterationsCount=config["MotionEstimation"]["iterationsCount"],
                                        reprojectionError=config["MotionEstimation"]["reprojectionError"],
                                        confidence=config["MotionEstimation"]["confidence"],
                                        flags=cv2.SOLVEPNP_P3P)
        

        rvec, tvec = cv2.solvePnPRefineLM(
            tracking_info["kpoints3d_prev"][inliers],
            tracking_info["kpoints2d_left_cur"][inliers],
            tracking_info["kl"],
            tracking_info["dist"],
            rvec, 
            tvec,
        )

        rmat, _ = cv2.Rodrigues(rvec)
        rmat = rmat.T
        tvec = -np.dot(rmat, tvec)
        return rmat, tvec


   