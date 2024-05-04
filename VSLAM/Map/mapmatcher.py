import numpy as np
import cv2
from VSLAM.Camera.stereocamera import StereoCamera
from typing import List
from ..utils import get_config, transform_points3d

config = get_config()


class MapMatcher:
    def __init__(self):
        self.matcher = cv2.BFMatcher()

    def match(self, cameras: List[StereoCamera], window: int):
        points3d = transform_points3d(cameras[-1].kpoints3d, cameras[-1].x)
        desc3d = cameras[-1].desc3d

        data_association = np.vstack(
            [
                self.match_camera(idx, points3d, desc3d, cameras[-window:])
                for idx in range(len(cameras[-window:]))
            ]
        )
        # ======================== DEBUGGING ===============================
        """
        idx = 0
        assoc = data_association[np.where(data_association[:, 1] == idx)]
        pts2d = cameras[-1].left_kp[assoc[:, 0]]
        matchd2d = cameras[-window:][idx].left_kp[assoc[:, 2]]

        matches = np.array(
            [
                cv2.DMatch(_queryIdx=idx, _trainIdx=idx, _imgIdx=0, _distance=0)
                for idx in range(len(pts2d))
            ]
        )
        sample_matches = np.random.randint(0, len(pts2d), size=(8,))
        matches = matches[sample_matches]

        stereo_matches = cv2.drawMatches(
            cameras[-1].left_image,
            pts2d,
            cameras[-window:][idx].left_image,
            matchd2d,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imshow("Matches", stereo_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("hi")
        """
        return data_association


    def transfom_points3d(self, points3d: np.ndarray, T: np.ndarray):
        points_homogeneous = np.hstack((points3d, np.ones((points3d.shape[0], 1))))
        transformed_points_homogeneous = points_homogeneous.dot(T.T)
        transformed_points = transformed_points_homogeneous[:, :3]
        return transformed_points

    def match_camera(
        self,
        camera_index: np.ndarray,
        points3d: np.ndarray,
        desc3d: np.ndarray,
        cameras: List[StereoCamera],
    ):
        """returns i, k, j
        i the 3d point index
        k the camera index
        j the 2d keypoint
        """
        camera = cameras[camera_index]

        matches = self.matcher.knnMatch(desc3d, camera.left_desc2d, k=2)

        queryidxs = np.array(
            [
                m.queryIdx
                for m, n in matches
                if m.distance < config["LoweRatio"] * n.distance or m.distance == 0.0
            ]
        )

        trainidxs = np.array(
            [
                m.trainIdx
                for m, n in matches
                if m.distance < config["LoweRatio"] * n.distance or m.distance == 0.0
            ]
        )

        matched2d = camera.left_kpoints2d[trainidxs]
        matched3d = points3d[queryidxs]
        obs2d = camera.project(matched3d)
        mask = self.filter_inliers2d(obs2d, matched2d, camera.kl)
        queryidxs = queryidxs[mask].reshape(-1, 1)
        trainidxs = trainidxs[mask].reshape(-1, 1)
        camera_indexs = np.full(len(trainidxs), camera_index).reshape(len(trainidxs), 1)
        return np.hstack((queryidxs, camera_indexs, trainidxs))

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

    def filter_inliers3d(self, points3d, inliers3d, inliers2d, camera):
        # filter by reprojection error