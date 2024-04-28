from .base import ABCFeatureTracker
from ...utils import get_config
import numpy as np
from typing import Tuple, List
from VSLAM.Features.Local import Describers
from VSLAM.utils import pts2kp
import cv2
from typing import Literal


class KLTTracker(ABCFeatureTracker):
    def __init__(self, lowe_ratio=0.75, ransac_threshold=5):
        self.ransac_threshold = ransac_threshold
        self.lowe_ratio = lowe_ratio
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )


    def filter_inliers2d(self, pts1, pts2):
        _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.ransac_threshold)
        return mask

    def __call__(self, camera1, camera2) -> Tuple:
                # Track features on left and right frame in the current state from previous states 
        _, pointsTrackedLeft, maskTrackingLeft = self.track_features(camera1.left_image,
                                                                               camera2.left_image,
                                                                               camera1.left_kpoints2d)

        _, pointsTrackedRight, maskTrackingRight = self.track_features(camera1.right_image,
                                                                     camera2.right_image,
                                                                     camera1.right_kpoints2d)
        ########### Need to filter the feature tracks using RANSAC HERE ################

        
        # Joint index and select only good tracked points
        points3d_left = camera1.points3d[maskTrackingLeft.flatten()==1]
        points3d_right = camera1.points3d[maskTrackingRight.flatten()==1]
        return (points3d_left, pointsTrackedLeft, points3d_right, pointsTrackedRight)
        
    

    def track_features(self, imageref, imagecur, ptsref):
        assert len(ptsref.shape) == 2 and ptsref.shape[1] == 2

        # Reshape input and track features
        ptsref = ptsref.reshape(-1, 1, 2).astype('float32')
        points_t0_t1, mask_t0_t1, _ = cv2.calcOpticalFlowPyrLK(imageref, 
                                                               imagecur,
                                                               ptsref, 
                                                               None, 
                                                               **self.lk_params)

        # Reshape ouput and return output and mask for tracking points
        ptsref = ptsref.reshape(-1,2)
        points_t0_t1 = points_t0_t1.reshape(-1,2)
        mask_t0_t1 = mask_t0_t1.flatten().astype(bool)

        return ptsref, points_t0_t1, mask_t0_t1