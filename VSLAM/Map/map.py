from typing import List, Tuple, Dict
from VSLAM.Camera.stereocamera import StereoCamera
from VSLAM.FeatureMatchers import FeatureMatcher
import numpy as np
from ..utils import get_config
from .mapmatcher import MapMatcher
import cv2
from ..utils import homogenize, transform_points3d
from scipy.optimize import least_squares


config = get_config()


def reprojection_cost(params, data_association, cameras, window_size):
    n_cameras = len(cameras[-window_size:])
    all_pts3d = params[n_cameras*6:].reshape(-1, 3)
    residuals = []
    for idx in range(n_cameras):
        rvec = params[idx * 6: idx *6 +3]
        tvec = params[idx*6+3:idx*6+6]

        da_idx = data_association[np.where(data_association[:, 1]==idx)]
        pts3d = all_pts3d[da_idx[:, 0]]
        obs2d = cameras[-window_size:][idx].left_kpoints2d[da_idx[:, 2]]

        pts2d = cv2.projectPoints(
                    pts3d,
                    cv2.Rodrigues(cameras[-window_size:][idx].rmat)[0],#rvec,
                    cameras[-window_size:][idx].tvec,#tvec,

                    cameras[-window_size:][idx].kl,
                    cameras[-window_size:][idx].dist)[0]
        residuals += np.abs(pts2d.flatten() - obs2d.flatten()).tolist()
    return np.array(residuals)



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
        self.count = 0

    def __call__(self, camera: StereoCamera):
        self.cameras.append(camera)
        self.count += 1

    def bundle_adjustment(self):
        data_association = self.map_matcher.match(self.cameras, self.window)
        params = cameras2params(self.cameras, self.window)
        residuals = reprojection_cost(params, data_association, self.cameras, self.window)
        print(np.mean(residuals))
        result = least_squares(
            reprojection_cost,
            params,
            args=(
                data_association,
                self.cameras,
                self.window
            ),
            method='lm'
        )
        print("finish optimize")
        import matplotlib.pyplot as plt
        bins = np.linspace(np.min(residuals), np.max(residuals), 100)
        plt.hist(residuals.flatten(), bins=bins)
        plt.title(str(self.count))
        plt.show()
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
