from typing import List, Tuple, Dict
from VSLAM.Camera.stereocamera import StereoCamera
from VSLAM.FeatureMatchers import FeatureMatcher
import numpy as np
from ..utils import get_config
from .mapmatcher import MapMatcher
import cv2
from ..utils import homogenize, transform_points3d


config = get_config()


def reprojection_cost(params, data_association, cameras):
    """

    :param params: a vector [rvec1, tvec1, rvec2, tv3c2, ..., x1, y1, z1, x2, y2, z2]
    :param data_association: an array of [i, k, j] the ith 3d point is matched with the jth observation of the kth camera
    :param cameras: the list of camera objects
    :return:
    """
    pass


def cameras2params(cameras: np.ndarray[StereoCamera], window_size: int):
    params = []
    for cam in cameras[-window_size:]:
        rvec = cv2.Rodrigues(cam.rmat)[0].flatten()
        tvec = cam.x[:3, 3]
        params += rvec.flatten().tolist()
        params += tvec.flatten().tolist()

    params += transform_points3d(cameras[-1].kpoints3d, cameras[-1].x).flatten().tolist()
    return np.array(params)


def params2cameras(
    params: np.ndarray,
    n_cameras: int,
    cameras: np.ndarray[StereoCamera],
    window_size: int,
):

    for idx in range(n_cameras):
        rvec = params[idx * 6 : idx * 6 + 3]
        tvec = params[idx * 6 + 3 : idx * 6 + 6]
        cameras[-window_size:][idx].x = homogenize(rvec, tvec)
        cameras[-window_size:][idx].rmat = cv2.Rodrigues(rvec)[0]
        cameras[-window_size:][idx].tvec = tvec
    points3d = params[n_cameras * 6 :].reshape(-1, 3)
    points3d = transform_points3d(points3d, np.linalg.inv(cameras[-1].x))
    cameras[-1].kpoints3d = points3d
    return cameras


class Map:
    def __init__(self):
        self.cameras = []
        self.window = config["Map"]["WindowSize"]
        self.map_matcher = MapMatcher()

    def __call__(self, camera: StereoCamera):
        self.cameras.append(camera)
        if config["LocalOptimization"].lower() == "bundleadjustment":
            self.bundle_adjustment()

    def bundle_adjustment(self):
        data_association = self.map_matcher.match(self.cameras, self.window)
        params = cameras2params(self.cameras, self.window)
        self.cameras = params2cameras(
            params,
            len(self.cameras[-self.window:]),
            self.cameras,
            self.window,
        )

    def local_map(self):
        return np.vstack([pt.kpoints3d for pt in self.cameras[-self.window :]])

    def global_map(self):
        return np.vstack([pt.kpoints3d for pt in self.cameras[-self.window :]])

    def local_traj(self):
        return np.vstack([pt.x[:3, 3] for pt in self.cameras[-self.window :]])

    def global_traj(self):
        return np.vstack([pt.x[:3, 3] for pt in self.cameras])
