import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from typing import List, Tuple

import numpy as np
import pytest

from Datasets import Kitti
from VSLAM.Camera import StereoCamera
from VSLAM.Features import LocalFeatures
from VSLAM.FeatureTrackers.Trackers import KLTTracker



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
    obj = KLTTracker()
    assert obj is not None 


def test_left_right_track_type():
    cam1, cam2 = get_cameras()
    lf = LocalFeatures()
    cam1 = lf.detectAndCompute(cam1)
    tracker = KLTTracker()
    cam1 = tracker.track(cam1)
    assert isinstance(cam1.left_kp, np.ndarray)
    assert isinstance(cam1.left_kpoints2d, np.ndarray)
    assert isinstance(cam1.right_kpoints2d, np.ndarray)


def test_left_right_track_size():
    cam1, cam2 = get_cameras()
    lf = LocalFeatures()
    cam1 = lf.detectAndCompute(cam1)
    tracker = KLTTracker()
    cam1 = tracker.track(cam1)
    assert len(cam1.left_kp) == len(cam1.right_kpoints2d)
    assert len(cam1.left_kpoints2d) == len(cam1.right_kpoints2d)
    assert len(cam1.left_kp) == len(cam1.left_kpoints2d)



def test_consecutive_type():
    cam1, cam2 = get_cameras()
    lf = LocalFeatures()
    cam1 = lf.detectAndCompute(cam1)
    cam2 = lf.detectAndCompute(cam2)
    tracker = KLTTracker()
    cam1, cam2 = tracker.track(cam1, cam2)
    assert isinstance(cam1.left_kp, np.ndarray)
    assert isinstance(cam1.left_kpoints2d, np.ndarray)
    assert isinstance(cam2.right_kpoints2d, np.ndarray)


def test_consecutive_size():
    cam1, cam2 = get_cameras()
    lf = LocalFeatures()
    cam1 = lf.detectAndCompute(cam1)
    cam2 = lf.detectAndCompute(cam2)
    tracker = KLTTracker()
    cam1, cam2 = tracker.track(cam1, cam2)
    assert len(cam1.left_kp) == len(cam2.left_kpoints2d)
    assert len(cam1.left_kpoints2d) == len(cam2.left_kpoints2d)
    assert len(cam1.left_kp) == len(cam2.left_kpoints2d)



