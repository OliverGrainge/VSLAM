from .base import ABCFeatureTracker
from ...utils import get_config
import numpy as np
from typing import Tuple, List
from VSLAM.Features.Local import Describers
from VSLAM.utils import pts2kp
import cv2


class KLTTracker(ABCFeatureTracker):
    def __init__(self, lowe_ratio=0.75):
        self.lowe_ratio = 0.75
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def track_left_to_right(self, camera1):
        left_pts = cv2.KeyPoint_convert(camera1.left_kp)
        right_pts, st, err = cv2.calcOpticalFlowPyrLK(
            camera1.left_image, camera1.right_image, left_pts, None, **self.lk_params
        )

        height, width = camera1.left_image.shape[:2]
        mask = (
            (st.flatten() == 1)
            & (right_pts[:, 0] >= 0)
            & (right_pts[:, 0] < width)
            & (right_pts[:, 1] >= 0)
            & (right_pts[:, 1] < height)
            & (np.abs(left_pts[:, 1] - right_pts[:, 1]) < 2.0) # limit movement in y direction
            & (np.abs(left_pts[:, 0] - right_pts[:, 0]) < 40) # limit movement in x direction
        )



        camera1.left_kp = camera1.left_kp[mask]
        camera1.left_kpoints2d = left_pts[mask]
        camera1.left_desc2d = camera1.left_desc2d[mask]

        camera1.right_kpoints2d = right_pts[mask]
        camera1.right_kp = pts2kp(camera1.right_kpoints2d)
        camera1.right_desc2d = None
        return camera1

    def track_consecutive(self, camera1, camera2) -> Tuple:
        prev_pts = cv2.KeyPoint_convert(camera1.left_kp)
        cur_pts, st, err = cv2.calcOpticalFlowPyrLK(
            camera1.left_image, camera2.left_image, prev_pts, None, **self.lk_params
        )

        height, width = camera1.left_image.shape[:2]
        mask = (
            (st.flatten() == 1) # those which klt assumes has good photometric error
            & (cur_pts[:, 0] >= 0) # new points tracked out the left of the image
            & (cur_pts[:, 0] < width) # new point tracked out the right of the image
            & (cur_pts[:, 1] >= 0) # new points tracked below the image
            & (cur_pts[:, 1] < height) # new points tracked above the image
            & (np.abs(cur_pts[:, 1] - prev_pts[:, 1]) < 15) # limit movement in y direction
            & (np.abs(cur_pts[:, 0] - prev_pts[:, 0]) < 15) # limit movement in x direction
        )


        camera1.left_kp = camera1.left_kp[mask]
        camera1.left_kpoints2d = prev_pts[mask]
        camera1.left_desc2d = camera1.left_desc2d[mask]
        camera1.kpoints3d = camera1.kpoints3d[mask]
        camera1.desc3d = camera1.left_desc2d

        camera2.left_kpoints2d = cur_pts[mask]
        camera2.left_kp = pts2kp(cur_pts)
        camera2.left_desc2d = None
        return camera1, camera2

    def track(self, camera1, camera2=None) -> Tuple:
        if camera2 is not None:
            return self.track_consecutive(camera1, camera2)
        else:
            return self.track_left_to_right(camera1)
            
