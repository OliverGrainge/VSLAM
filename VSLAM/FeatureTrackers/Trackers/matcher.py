from typing import List, Tuple

import cv2
import numpy as np

from .base import ABCFeatureTracker


class MatcherTracker(ABCFeatureTracker):
    def __init__(self, lowe_ratio=0.75, ransac_reproj_threshold=5.0):
        self.lowe_ratio = lowe_ratio
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.binary_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.dense_matcher = cv2.BFMatcher()

    def apply_ransac(self, kp1, kp2, matches):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.ransac_reproj_threshold)
        return mask

    def track_left_to_right(self, camera1):
        if camera1.left_desc2d.dtype == np.float32:
            matches = self.dense_matcher.knnMatch(
                camera1.right_desc2d, camera1.left_desc2d, k=2
            )
        else:
            matches = self.binary_matcher.knnMatch(
                camera1.right_desc2d, camera1.left_desc2d, k=2
            )
        # use lowes ratio test
        matches = [m for m, n in matches if m.distance < self.lowe_ratio * n.distance]
        mask = self.apply_ransac(camera1.right_kp, camera1.left_kp, matches)
        matches = [matches[i] for i in range(len(mask)) if mask[i]]

        new_kp_ref = []
        new_kp_cur = []
        new_desc_ref = np.zeros(
            (len(matches), camera1.left_desc2d.shape[1]),
            dtype=camera1.left_desc2d.dtype,
        )
        new_desc_cur = np.zeros(
            (len(matches), camera1.right_desc2d.shape[1]),
            dtype=camera1.right_desc2d.dtype,
        )

        for idx, match in enumerate(matches):
            new_desc_cur[idx] = camera1.right_desc2d[match.queryIdx]
            new_desc_ref[idx] = camera1.left_desc2d[match.trainIdx]
            new_kp_cur.append(camera1.right_kp[match.queryIdx])
            new_kp_ref.append(camera1.left_kp[match.trainIdx])


        camera1.left_kp = np.array(new_kp_ref)
        camera1.left_kpoints2d = cv2.KeyPoint_convert(new_kp_ref)
        camera1.left_desc2d = new_desc_ref

        camera1.right_kp = np.array(new_kp_cur)
        camera1.right_kpoints2d = cv2.KeyPoint_convert(new_kp_cur)
        camera1.right_desc2d = new_desc_cur
        return camera1

    def track_consecutive(self, camera1, camera2):
        if camera1.left_desc2d.dtype == np.float32:
            matches = self.dense_matcher.knnMatch(
                camera2.left_desc2d, camera1.left_desc2d, k=2
            )
        else:
            matches = self.binary_matcher.knnMatch(
                camera2.left_desc2d, camera1.left_desc2d, k=2
            )
        # use lowes ratio test
        matches = [m for m, n in matches if m.distance < self.lowe_ratio * n.distance]
        mask = self.apply_ransac(camera2.left_kp, camera1.left_kp, matches)
        matches = [matches[i] for i in range(len(mask)) if mask[i]]

        new_kp_ref = []
        new_kp_cur = []
        new_desc_ref = np.zeros(
            (len(matches), camera1.left_desc2d.shape[1]),
            dtype=camera1.left_desc2d.dtype,
        )
        new_desc_cur = np.zeros(
            (len(matches), camera2.left_desc2d.shape[1]),
            dtype=camera2.left_desc2d.dtype,
        )

        new_pts3d = np.zeros(
            (len(matches), 3)
        )

        
        for idx, match in enumerate(matches):
            new_pts3d[idx] = camera1.kpoints3d[match.trainIdx]
            new_desc_cur[idx] = camera2.left_desc2d[match.queryIdx]
            new_desc_ref[idx] = camera1.left_desc2d[match.trainIdx]
            new_kp_cur.append(camera2.left_kp[match.queryIdx])
            new_kp_ref.append(camera1.left_kp[match.trainIdx])


        camera1.left_kp = np.array(new_kp_ref)
        camera1.left_desc2d = new_desc_ref
        camera1.left_kpoints2d = cv2.KeyPoint_convert(new_kp_ref)
        camera1.kpoints3d = new_pts3d
        camera1.desc3d = new_desc_ref

        camera2.left_kp = np.array(new_kp_cur)
        camera2.left_kpoints2d = cv2.KeyPoint_convert(new_kp_cur)
        camera2.left_desc2d = new_desc_cur
        return camera1, camera2

    def track(
        self,
        camera1,
        camera2=None,
    ) -> Tuple[List, np.ndarray, List, np.ndarray]:
        if camera2 is None:
            return self.track_left_to_right(camera1)
        else:
            return self.track_consecutive(camera1, camera2)
