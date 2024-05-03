import numpy as np
import cv2
from VSLAM.Camera.stereocamera import StereoCamera
from typing import List
from ..utils import get_config

config = get_config()

class MapMatcher:
    def __init__(self):
        self.matcher = cv2.BFMatcher()

    def match(
        self,
        cameras: List[StereoCamera],
    ):
        """
        This function takes in the 3d points and descriptors of the points. Matches those points
        with the camera observations and returns a list of (i, k, j) where the ith 3d point is
        matched to the jth observation in the kth camera.

        :param points3D:
        :param desc3d:
        :param camera:
        :return:
        """
        points3d = np.vstack([self.transfom_points3d(cam.kpoints3d, cam.x) for cam in cameras])
        desc3d = np.vstack([cam.desc3d for cam in cameras])

        data_association = np.vstack([
            self.match_camera(idx,
                              points3d,
                              desc3d,
                              self.cameras[-self.window:]) for idx in range(len(self.cameras[-self.window:])))
        return data_association


    def transfom_points3d(self, points3d: np.ndarry, T: np.ndarray):
        points_homogeneous = np.hstack((points3d, np.ones((points3d.shape[0], 1))))
        transformed_points_homogeneous = points_homogeneous.dot(T.T)
        transformed_points = transformed_points_homogeneous[:, :3]
        return transformed_points


    def match_camera(self,
        camera_index: np.ndarray,
        points3d: np.ndarray,
        desc3d: np.ndarray,
        cameras: List[StereoCamera],
    ):
        """ returns i, k, j
        i the 3d point index
        k the camera index
        j the 2d keypoint
        """
        camera = cameras[camera_index]
        matches = self.matcher.knn(camera.left_desc2d, desc3d, k=2)
        queryidxs = [m.queryIdx for m, n in matches if m.distance < config["LoweRatio"] * n.distance]
        trainidxs = [m.trainIdx for m, n in matches if m.distance < config["LoweRatio"] * n.distance]

        matched2d = camera.left_kpoints2d[queryidxs]
        matched3d = points3d[trainidxs]
        obs2d = camera.project(matched3d)

        mask = self.filter_inliers2d(matched2d, obs2d, camera.kl)
        queryidxs = queryidxs[mask].reshape(-1, 1)
        trainidxs = trainidxs[mask].reshape(-1, 1)
        camera_indexs = np.full(len(trainidxs), camera_index).reshape(len(trainidxs), 1)
        return np.hstack((trainidxs, camera_indexs, queryidxs))


    def filter_inliers2d(self, pts1, pts2, k):
        _, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            k,
            method=8,
            prob=config["Map"]["BundleAdjustment"]["Probability"],
            threshold=config["Map"]["BundleAdjustment"]["InlierThreshold"],
        )

        mask = mask.ravel().astype(bool)
        return mask


