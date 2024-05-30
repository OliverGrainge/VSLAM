from typing import List, Tuple, Dict
from VSLAM.Camera.stereocamera import StereoCamera
from VSLAM.FeatureMatchers import FeatureMatcher
import numpy as np
from ..utils import get_config
from .mapmatcher import MapMatcher
import cv2
from ..utils import homogenize, transform_points3d, pts2kp
from scipy.optimize import least_squares
from ..Backend import data_assocation, reprojection_error, params2cameras, cameras2params, get_obs_from_cameras, ba_cost

config = get_config()

class Map:
    def __init__(self):
        super().__init__()
        self.cameras = []
        self.window = config["Map"]["WindowSize"]
        self.count = 0

    def __call__(self, camera: StereoCamera, tracking_info):
        self.cameras.append(camera)
        
        if len(self.cameras) >= self.window: 
            if config["LocalOptimization"] == "BundleAdjustment":
                params, n_cameras = cameras2params(self.cameras, self.window)
                params2cameras(params, self.cameras, self.window)
                da = data_assocation(self.cameras, self.window)
                obs = get_obs_from_cameras(self.cameras, self.window)
                kl = self.cameras[-1].kl
                cost_before = np.abs(ba_cost(params, n_cameras, np.zeros(5), kl, obs, da))
                result = least_squares(
                    ba_cost, 
                    params, 
                    args=(
                        n_cameras,
                        np.zeros(5),
                        kl,
                        obs, 
                        da
                    ),
                    #method='lm',
                    ftol=1e-3, 
                    xtol=1e-3, 
                    gtol=1e-3,
                    loss='huber',
                    f_scale=2.5
                )
                cost_after = np.abs(ba_cost(result.x, n_cameras, np.zeros(5), kl, obs, da))
                params2cameras(result.x, self.cameras, self.window)
                print("cost_before", np.mean(cost_before), "cost_after", np.mean(cost_after))

        self.count += 1

    def local_map(self):
        return np.vstack([pt.kpoints3d for pt in self.cameras[-self.window :]])

    def global_map(self):
        return np.vstack([pt.kpoints3d for pt in self.cameras[-self.window :]])

    def local_traj(self):
        return np.vstack([pt.x[:3, 3] for pt in self.cameras[-self.window :]])

    def global_traj(self):
        return np.vstack([pt.x[:3, 3] for pt in self.cameras])
 