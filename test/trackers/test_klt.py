import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)


import numpy as np
from Datasets import Kitti
from VSLAM.Camera import StereoCamera
from VSLAM.FeatureTrackers import FeatureTracker


def get_cameras():
    root = os.path.join(os.getcwd(), "/test/data/")[1:]
    ds = Kitti(root=root)
    inputs1 = ds.load_frame(0)
    inputs2 = ds.load_frame(1)
    params = ds.load_parameters()
    camera1 = StereoCamera(**inputs1, **params)
    camera2 = StereoCamera(**inputs2, **params)
    return camera1, camera2


def test_instantiation():
    tracker = FeatureTracker()
    assert tracker is not None

def test_tracking_type():
    cam1, cam2 = get_cameras()
    tracker = FeatureTracker()
    tracking_info = tracker.track(cam1, cam2)
    assert isinstance(tracking_info, dict)
    assert isinstance(tracking_info["kpoints3d_prev"], np.ndarray)
    assert isinstance(tracking_info["kpoints2d_left_prev"], np.ndarray)
    assert isinstance(tracking_info["kpoints2d_right_prev"], np.ndarray)
    assert isinstance(tracking_info["kpoints2d_left_cur"], np.ndarray)
    assert isinstance(tracking_info["kpoints2d_right_cur"], np.ndarray)
    assert isinstance(tracking_info["kp_left_cur"], np.ndarray)
    assert isinstance(tracking_info["kp_right_cur"], np.ndarray)
    assert isinstance(tracking_info["kp_left_prev"], np.ndarray)
    assert isinstance(tracking_info["kp_right_prev"], np.ndarray)
    assert isinstance(tracking_info["kp_left_cur"], np.ndarray)
    assert isinstance(tracking_info["kl"], np.ndarray)
    assert isinstance(tracking_info["kr"], np.ndarray)
    assert isinstance(tracking_info["dist"], np.ndarray)
    assert isinstance(tracking_info["pl"], np.ndarray)
    assert isinstance(tracking_info["pr"], np.ndarray)



def test_tracking_shape2():
    cam1, cam2 = get_cameras()
    tracker = FeatureTracker()
    tracking_info = tracker.track(cam1, cam2)
    n_points = len(tracking_info["kpoints3d_prev"])
    assert n_points > 30

    assert tracking_info["kpoints3d_prev"].shape[0] == n_points
    assert tracking_info["kpoints2d_left_prev"].shape[0] == n_points
    assert tracking_info["kpoints2d_right_prev"].shape[0] == n_points
    assert tracking_info["kpoints2d_left_cur"].shape[0] == n_points
    assert tracking_info["kpoints2d_right_cur"].shape[0] == n_points
    assert tracking_info["kp_left_cur"].shape[0] == n_points
    assert tracking_info["kp_right_cur"].shape[0] == n_points
    assert tracking_info["kp_left_prev"].shape[0] == n_points
    assert tracking_info["kp_right_prev"].shape[0] == n_points
    assert tracking_info["kp_left_cur"].shape[0] == n_points
    assert tracking_info["kl"].shape == (3, 3)
    assert tracking_info["kr"].shape == (3, 3)
    assert tracking_info["dist"].shape == (5,)
    assert tracking_info["pl"].shape == (3, 4)
    assert tracking_info["pr"].shape == (3, 4)
