import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from typing import List, Tuple

import numpy as np
import pytest

from Datasets import Kitti
from VSLAM.Camera import StereoCamera
from VSLAM.MotionEstimation.motion3d2d import MotionEstimation3D2D
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
    estimate = MotionEstimation3D2D()
    assert estimate is not None

def test_estimation():
    cam1, cam2 = get_cameras()
    me = MotionEstimation3D2D()
    tracking = FeatureTracker()
    tracking_info = tracking.track(cam1, cam2)
    rmat, tvec = me(tracking_info)
    assert rmat.shape == (3,3)
    assert tvec.shape == (3, 1)





