from typing import List, Tuple, Dict
from VSLAM.Camera.stereocamera import StereoCamera
from VSLAM.FeatureMatchers import FeatureMatcher
import numpy as np
from ..utils import get_config
from .mapmatcher import MapMatcher

config = get_config()



def reprojection_cost(params, data_association, cameras):
    """

    :param params: a vector [rvec1, tvec1, rvec2, tv3c2, ..., x1, y1, z1, x2, y2, z2]
    :param data_association: an array of [i, k, j] the ith 3d point is matched with the jth observation of the kth camera
    :param cameras: the list of camera objects
    :return:
    """
    pass

def cameras2params(params):
    pass

def params2cameras(params, n_cameras):
    pass

class Map:
    def __init__(self):
        self.cameras = []
        self.window = config["Map"]["WindowSize"]
        self.map_matcher = MapMatcher()

    def __call__(self, camera: StereoCamera):
        self.cameras.append(camera)
        if config["BundleAdjust"]:
            NotImplementedError


    def bundle_adjustment(self):
        data_association = self.map_matcher.match(self, self.cameras)


    def local_map(self):
        return np.vstack([pt.kpoints3d for pt in self.cameras[-self.window:])

    def global_map(self):
        return np.vstack([pt.kpoints3d for pt in self.cameras[-self.window:]])

    def local_traj(self):
        return np.vstack([pt.x[:3, 3] for pt in self.cameras[-self.window:]])

    def global_traj(self):
        return np.vstack([pt.x[:3, 3] for pt in self.cameras])





