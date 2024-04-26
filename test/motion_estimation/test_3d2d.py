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
from VSLAM.Features import LocalFeatures
from VSLAM.FeatureTrackers import FeatureTracker



def get_data_consecutive():
    root = os.path.join(os.getcwd(), "/test/data/")[1:]
    ds = Kitti(root=root)
    inputs1 = ds.load_frame(0)
    inputs2 = ds.load_frame(1)
    params = ds.load_parameters()
    cam1= StereoCamera(**inputs1, **params)
    cam2 = StereoCamera(**inputs2, **params)
    tracker = FeatureTracker()
    lf = LocalFeatures()
    cam1 = lf.detectAndCompute(cam1)
    cam2 = lf.detectAndCompute(cam2)
    cam1 = tracker.track(cam1)
    cam1.triangulate()
    pos = ds.ground_truth()[1]
    return cam1, cam2, pos

def test_instantiation():
    estimate = MotionEstimation3D2D()
    assert estimate is not None

def test_estimation():
    cam1, cam2, gt_pose = get_data_consecutive()
    estimate = MotionEstimation3D2D()
    tracker = FeatureTracker()
    cam1, cam2 = tracker.track(cam1, cam2)
    T = estimate(cam1, cam2)
    tvec = T[:3, 3]
    ttarg = gt_pose[:3, 3]
    print(tvec)
    print(ttarg)
    raise Exception





