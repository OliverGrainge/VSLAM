from .base import ABCFeatureTracker
import numpy as np
from typing import Tuple
import cv2
from ...utils import pts2kp, get_config

config = get_config()


class KLTTracker(ABCFeatureTracker):
    def __init__(
        self,
    ):
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    @staticmethod
    def filter_matching_inliers(leftMatchesPoints, rightMatchedPoints, intrinsic):
        _, mask_epipolar = cv2.findEssentialMat(
            leftMatchesPoints,
            rightMatchedPoints,
            intrinsic,
            method=8,
            threshold=config["Tracking"]["InlierThreshold"],
            prob=config["Tracking"]["Probability"],
        )

        mask_epipolar = mask_epipolar.ravel().astype(bool)
        return mask_epipolar

    def track(self, camera1, camera2) -> Tuple:
        _, pointsTrackedLeft, maskTrackingLeft = self.track_features(
            camera1.left_image, camera2.left_image, camera1.left_kpoints2d
        )

        _, pointsTrackedRight, maskTrackingRight = self.track_features(
            camera1.right_image, camera2.right_image, camera1.right_kpoints2d
        )

        tracking_info = {}
        mask = np.logical_and(
            maskTrackingLeft.flatten(), maskTrackingRight.flatten()
        ).astype(bool)
        tracking_info["kpoints3d_prev"] = camera1.kpoints3d[mask]
        tracking_info["kpoints2d_left_cur"] = pointsTrackedLeft[mask]
        tracking_info["kpoints2d_right_cur"] = pointsTrackedRight[mask]
        tracking_info["kpoints2d_left_prev"] = camera1.left_kpoints2d[mask]
        tracking_info["kpoints2d_right_prev"] = camera1.right_kpoints2d[mask]
        tracking_info["kp_left_cur"] = pts2kp(pointsTrackedLeft[mask])
        tracking_info["kp_right_cur"] = pts2kp(pointsTrackedRight[mask])
        tracking_info["kp_left_prev"] = camera1.left_kp[mask]
        tracking_info["kp_right_prev"] = camera1.right_kp[mask]
        mask_left = self.filter_matching_inliers(
            tracking_info["kpoints2d_left_prev"],
            tracking_info["kpoints2d_left_cur"],
            camera1.kl,
        )
        mask_right = self.filter_matching_inliers(
            tracking_info["kpoints2d_right_prev"],
            tracking_info["kpoints2d_right_cur"],
            camera1.kr,
        )

        mask = np.logical_and(mask_left, mask_right)

        tracking_info["kpoints3d_prev"] = tracking_info["kpoints3d_prev"][mask]
        tracking_info["kpoints2d_left_cur"] = tracking_info["kpoints2d_left_cur"][mask]
        tracking_info["kpoints2d_left_prev"] = tracking_info["kpoints2d_left_prev"][
            mask
        ]
        tracking_info["kpoints2d_right_cur"] = tracking_info["kpoints2d_right_cur"][
            mask
        ]
        tracking_info["kpoints2d_right_prev"] = tracking_info["kpoints2d_right_prev"][
            mask
        ]
        tracking_info["kp_left_cur"] = tracking_info["kp_left_cur"][mask]
        tracking_info["kp_right_cur"] = tracking_info["kp_right_cur"][mask]
        tracking_info["kp_left_prev"] = tracking_info["kp_left_prev"][mask]
        tracking_info["kp_right_prev"] = tracking_info["kp_right_prev"][mask]

        tracking_info["kl"] = camera2.kl
        tracking_info["kr"] = camera2.kr
        tracking_info["dist"] = camera2.dist
        tracking_info["pl"] = camera2.pl
        tracking_info["pr"] = camera2.pr
        return tracking_info

    def track_features(self, imageref, imagecur, ptsref):
        assert (
            len(ptsref.shape) == 2 and ptsref.shape[1] == 2
        ), "Input points are not in the correct shape."
        ptsref = ptsref.reshape(-1, 1, 2).astype("float32")
        points_t0_t1, mask_t0_t1, _ = cv2.calcOpticalFlowPyrLK(
            imageref, imagecur, ptsref, None, **self.lk_params
        )

        if points_t0_t1 is not None:
            points_t0_t1 = points_t0_t1.reshape(-1, 2)
            h, w = imagecur.shape[:2]

            within_bounds = (
                (points_t0_t1[:, 0] >= 0)
                & (points_t0_t1[:, 0] < w)
                & (points_t0_t1[:, 1] >= 0)
                & (points_t0_t1[:, 1] < h)
            )
            mask_t0_t1 = mask_t0_t1.flatten().astype(bool) & within_bounds
        else:
            points_t0_t1 = np.zeros_like(ptsref).reshape(-1, 2)
            mask_t0_t1 = np.zeros(ptsref.shape[0], dtype=bool)

        ptsref = ptsref.reshape(-1, 2)
        return ptsref, points_t0_t1, mask_t0_t1
