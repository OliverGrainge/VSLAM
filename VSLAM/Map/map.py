from typing import List, Tuple, Dict
from VSLAM.Camera.stereocamera import StereoCamera
from VSLAM.FeatureMatchers import FeatureMatcher
import numpy as np
from ..utils import get_config
from .mapmatcher import MapMatcher

config = get_config()

class Map:
    def __init__(self):
        self.cameras: List[StereoCamera] = []
        self.points3D: np.ndarray = np.array([])  # Nx3 array of 3D points
        self.desc3D: np.ndarray
        self.cameraIdx: np.ndarray = np.ndarray([])
        self.observations: Dict[int, List[Tuple[int, int]]] = {} # Maps 3D point index to a list of (camera_index, keypoint_index) tuples
        self.window_size = config["Map"]["WindowSize"]
        self.max_points_per_frame = config["Map"]["MaxPoints"] // self.window_size

        self.map_matcher = MapMatcher()


    def __call__(self):
        new_camera = self.cameras[-1]
        self.points3d = np.vstack(self.points3d, new_camera.kpoints3d[:self.max_points_per_frame])
        idxs = np.zeros(len(new_camera.kpoints3d[:self.max_points_per_frame]))
        self.cameraIdx = np.concatenate((
            self.cameraIdx,
            idxs.fill(np.max(self.cameraIdx) + 1)
        ))

        if np.max(self.cameraIdx) > len(self.window_size):
            idx = np.where(self.cameraIdx == 1)[0][0]
            self.cameraIdx = self.cameraIdx[idx:]
            self.cameraIdx -= 1
            self.points3D = self.points3D[idx:]





