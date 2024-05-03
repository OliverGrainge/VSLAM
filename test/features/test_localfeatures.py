import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)


import numpy as np
from Datasets import Kitti
from VSLAM.Camera import StereoCamera
from VSLAM.Features import LocalFeatures


def get_camera():
    root = os.path.join(os.getcwd(), "/test/data/")[1:]
    ds = Kitti(root=root)
    inputs = ds.load_frame(0)
    params = ds.load_parameters()
    camera = StereoCamera(**inputs, **params)
    return camera


def test_instantiation():
    lf = LocalFeatures()
    assert lf is not None


def test_detection():
    cam = get_camera()
    lf = LocalFeatures()
    cam = lf.detect(cam)
    assert len(cam.left_kp) > 10
    assert isinstance(cam.left_kp, np.ndarray)
    assert len(cam.right_kp) > 10
    assert isinstance(cam.right_kp, np.ndarray)


def test_desc():
    cam = get_camera()
    lf = LocalFeatures()
    cam = lf.compute(cam)
    assert len(cam.left_desc2d) > 10
    assert isinstance(cam.left_desc2d, np.ndarray)
    assert len(cam.left_desc2d) > 10
    assert isinstance(cam.left_desc2d, np.ndarray)


def test_detectandcompute():
    cam = get_camera()
    lf = LocalFeatures()
    cam = lf.detectAndCompute(cam)
    assert len(cam.left_desc2d) > 10
    assert isinstance(cam.left_desc2d, np.ndarray)
    assert len(cam.left_desc2d) > 10
    assert isinstance(cam.left_desc2d, np.ndarray)
