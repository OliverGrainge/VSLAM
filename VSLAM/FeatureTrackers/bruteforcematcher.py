from .base import ABCFeatureTracker
from ..utils import get_config 
import cv2
import numpy as np
from typing import Tuple, List

config = get_config()


class BruteForceTracker(ABCFeatureTracker):
    def __init__(self, lowe_ratio=0.75):
        self.lowe_ratio = 0.75
        self.binary_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.dense_matcher = cv2.BFMatcher()

    def track(
            self,
            image_ref: np.ndarray,
            image_cur: np.ndarray,
            kp_ref: List,
            desc_ref: np.ndarray, 
            kp_cur: List,
            desc_cur: np.ndarray,
        ) -> Tuple[List, np.ndarray, List, np.ndarray]:

        if desc_cur.dtype == np.float32:
            matches = self.dense_matcher.knnMatch(desc_cur, desc_ref, k=2)
        else: 
            matches = self.binary_matcher.knnMatch(desc_cur, desc_ref, k=2)
        # use lowes ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.lowe_ratio * n.distance:
                good_matches.append(m)
        matches = good_matches

        new_kp_ref = []
        new_kp_cur = []
        new_desc_ref = np.zeros((len(matches), desc_ref.shape[1]), dtype=desc_ref.dtype)
        new_desc_cur = np.zeros((len(matches), desc_cur.shape[1]), dtype=desc_cur.dtype)

        for idx, match in enumerate(matches):
            new_desc_cur[idx] = desc_cur[match.queryIdx]
            new_desc_ref[idx] = desc_ref[match.trainIdx]
            new_kp_cur.append(kp_cur[match.queryIdx])
            new_kp_ref.append(kp_ref[match.trainIdx])

        return (
            new_kp_ref,
            new_kp_cur,
            new_desc_ref,
            new_desc_cur,
        )



class BruteForceTracker(ABCFeatureTracker):
    def __init__(self, lowe_ratio=0.75):
        self.lowe_ratio = 0.75
        self.binary_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.dense_matcher = cv2.BFMatcher()

    def track(
            self,
            camera1, 
            camera2
        ) -> Tuple[List, np.ndarray, List, np.ndarray]:

        if camera1.desc2d.dtype == np.float32:
            matches = self.dense_matcher.knnMatch(camera2.desc2d, camera1.desc2d, k=2)
        else: 
            matches = self.binary_matcher.knnMatch(camera2.desc2d, camera1.desc2df, k=2)
        # use lowes ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.lowe_ratio * n.distance:
                good_matches.append(m)
        matches = good_matches

        new_kp_ref = []
        new_kp_cur = []
        new_desc_ref = np.zeros((len(matches), camera1.desc2d.shape[1]), dtype=camera1.desc2d.dtype)
        new_desc_cur = np.zeros((len(matches), camera2.desc2d.shape[1]), dtype=camera2.desc2d.dtype)

        for idx, match in enumerate(matches):
            new_desc_cur[idx] = camera2.desc2d[match.queryIdx]
            new_desc_ref[idx] = camera1.desc2d[match.trainIdx]
            new_kp_cur.append(camera2.kp[match.queryIdx])
            new_kp_ref.append(camera1.kp[match.trainIdx])

        camera1.kp = new_kp_ref,
        camera2.kp = new_kp_cur
        camera1.desc2d = new_desc_ref
        camera2.desc2d = new_desc_cur
        return camera1, camera2