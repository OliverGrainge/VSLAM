from .base import ABCFeatureTracker
from ..utils import get_config 
import cv2
import numpy as np
from typing import Tuple, List
from VSLAM.Features.Local import Describers
from VSLAM.utils import pts2kp

config = get_config()

def get_matcher(): 
    if config["LocalFeatureDescriber"] in ["sift"]:
        return cv2.BFMatcher()
    else: 
        return cv2.BFMatcher(cv2.NORM_HAMMING)


def get_describer(): 
    if config["LocalFeatureDescriber"] == "SIFT":
        return Describers.SIFT()
    else: 
        raise NotImplementedError()


class KLTTracker(ABCFeatureTracker):
    def __init__(self, lowe_ratio=0.75):
        self.lowe_ratio = 0.75
        self.lk_params = dict(
            winSize=(21, 21), 
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        ) 
        self.describer = get_describer()

    def track(
            self,
            image_ref: np.ndarray,
            image_cur: np.ndarray,
            kp_ref: List,
            desc_ref: np.ndarray, 
            kp_cur: List,
            desc_cur: np.ndarray,
        ) -> Tuple[List, np.ndarray, List, np.ndarray]:

        pts_ref = cv2.KeyPoint_convert(kp_ref)
        pts_cur, st, err = cv2.calcOpticalFlowPyrLK(
            image_ref, 
            image_cur, 
            pts_ref, 
            None, 
            **self.lk_params)

        height, width = image_cur.shape[:2]
        mask = (st.flatten() == 1) & (pts_cur[:, 0] >= 0) & (pts_cur[:, 0] < width) & (pts_cur[:, 1] >= 0) & (pts_cur[:, 1] < height)
    

        new_pts_ref = pts_ref[mask]
        new_pts_cur = pts_cur[mask]
        new_kp_ref = pts2kp(new_pts_ref)
        new_kp_cur = pts2kp(new_pts_cur)
        new_desc_ref = desc_ref[mask]


        return (
            new_kp_ref,
            new_desc_ref,
            new_kp_cur,
            None,
        )
