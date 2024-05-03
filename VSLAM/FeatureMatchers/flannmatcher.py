import cv2
import numpy as np
from ..utils import get_config
from .base import ABCFeatureMatcher

config = get_config()


class FlannMatcher(ABCFeatureMatcher):
    def __init__(self):
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=0, trees=5), dict(checks=50)
        )

    def match(self, camera):
        camera = self.get_matches(camera)
        camera = self.filter_inliers2d(camera)
        camera = self.triangulate(camera)
        camera = self.filter_inliers3d(camera)
        return camera

    def get_matches(self, camera):
        matches = self.matcher.knnMatch(camera.left_desc2d, camera.right_desc2d, 2)

        queryidxs = [m.queryIdx for m, n in matches if m.distance < config["LoweRatio"] * n.distance]
        trainidxs = [m.trainIdx for m, n in matches if m.distance < config["LoweRatio"] * n.distance]

        camera.left_kpoints2d = camera.left_kpoints2d[queryidxs]
        camera.right_kpoints2d = camera.right_kpoints2d[trainidxs]

        camera.left_kp = camera.left_kp[queryidxs]
        camera.right_kp = camera.right_kp[trainidxs]
        camera.left_desc2d = camera.left_desc2d[queryidxs]
        camera.right_desc2d = camera.right_desc2d[trainidxs]
        return camera

    def filter_inliers2d(self, camera):
        _, mask = cv2.findEssentialMat(
            camera.left_kpoints2d,
            camera.right_kpoints2d,
            camera.kl,
            method=8,
            prob=config["Stereo"]["Probability"],
            threshold=config["Stereo"]["InlierThreshold"],
        )

        mask = mask.ravel().astype(bool)
        camera.left_kp = camera.left_kp[mask]
        camera.right_kp = camera.right_kp[mask]
        camera.left_desc2d = camera.left_desc2d[mask]
        camera.right_desc2d = camera.right_desc2d[mask]
        camera.left_kpoints2d = camera.left_kpoints2d[mask]
        camera.right_kpoints2d = camera.right_kpoints2d[mask]
        return camera

    def triangulate(self, camera):
        points4d = cv2.triangulatePoints(
            camera.pl,
            camera.pr,
            camera.left_kpoints2d.T,
            camera.right_kpoints2d.T,
        )

        camera.kpoints3d = (points4d[:3, :] / points4d[3, :]).T
        camera.desc3d = camera.left_desc2d

        return camera

    def filter_inliers3d(self, camera):
        proj2d_left = cv2.projectPoints(
            camera.kpoints3d, np.zeros(3), np.zeros(3), camera.kl, camera.dist
        )[0].squeeze()

        proj2d_right = cv2.projectPoints(
            camera.kpoints3d,
            np.zeros(3),
            camera.baseline_vector,
            camera.kl,
            camera.dist,
        )[0].squeeze()

        reproj_error = (
            (np.sqrt(((proj2d_left - camera.left_kpoints2d) ** 2).sum(axis=1)))
            + (np.sqrt(((proj2d_right - camera.right_kpoints2d) ** 2).sum(axis=1)))
        ) / 2
        mask_x = np.logical_and(
            (camera.kpoints3d[:, 0] > config["Stereo"]["min_x"]),
            (camera.kpoints3d[:, 0] < config["Stereo"]["max_x"]),
        )
        mask_y = np.logical_and(
            (camera.kpoints3d[:, 1] < config["Stereo"]["max_y"]),
            (camera.kpoints3d[:, 1] > config["Stereo"]["min_y"]),
        )
        mask_z = camera.kpoints3d[:, 2] > config["Stereo"]["min_z"]
        mask_reproj = reproj_error < config["Stereo"]["MaxReprojError"]
        mask = np.logical_and(np.logical_and(mask_x, mask_y, mask_z), mask_reproj)

        camera.kpoints3d = camera.kpoints3d[mask]
        camera.left_kp = camera.left_kp[mask]
        camera.right_kp = camera.right_kp[mask]
        camera.left_desc2d = camera.left_desc2d[mask]
        camera.right_desc2d = camera.right_desc2d[mask]
        camera.left_kpoints2d = camera.left_kpoints2d[mask]
        camera.right_kpoints2d = camera.right_kpoints2d[mask]
        camera.desc3d = camera.desc3d[mask]
