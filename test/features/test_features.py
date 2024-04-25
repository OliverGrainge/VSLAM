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
from VSLAM.Features import LocalFeatures


def get_camera():
    root = os.path.join(os.getcwd(), "/test/data/")[1:]
    ds = Kitti(root=root)
    inputs = ds.load_frame(0)
    params = ds.load_parameters()
    return StereoCamera(**inputs, **params)


def test_instantiation():
    obj = LocalFeatures()
    assert obj is not None


def test_detect():
    lf = LocalFeatures()
    camera = get_camera()
    camera = lf.detect(camera)
    assert camera.left_kp is not None
    assert camera.right_kp is not None
    assert len(camera.left_kp) > 0
    assert len(camera.right_kp) > 0
    assert len(camera.right_kpoints2d) > 0
    assert len(camera.left_kpoints2d) > 0


def test_compute():
    lf = LocalFeatures()
    camera = get_camera()
    camera = lf.detect(camera)
    camera = lf.compute(camera)
    assert camera.left_desc2d is not None
    assert camera.right_desc2d is not None


def test_compute_dtype():
    lf = LocalFeatures()
    camera = get_camera()
    camera = lf.detect(camera)
    camera = lf.compute(camera)
    assert isinstance(camera.left_desc2d, np.ndarray)
    assert isinstance(camera.right_desc2d, np.ndarray)


def test_detectandcompute():
    lf = LocalFeatures()
    camera = get_camera()
    camera = lf.detectAndCompute(camera)
    assert isinstance(camera.left_kp, np.ndarray)
    assert isinstance(camera.right_kp, np.ndarray)
    assert isinstance(camera.right_desc2d, np.ndarray)
    assert isinstance(camera.left_desc2d, np.ndarray)
    assert len(camera.left_desc2d) == len(camera.left_kp) == len(camera.left_kpoints2d)
    assert (
        len(camera.right_desc2d) == len(camera.right_kp) == len(camera.right_kpoints2d)
    )
