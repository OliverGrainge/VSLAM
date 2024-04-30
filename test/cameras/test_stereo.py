import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import List, Tuple

import numpy as np
import pytest

from Datasets import Kitti
from VSLAM.Camera import StereoCamera


def get_dataset():
    root = os.path.join(os.getcwd(), "/test/data/")[1:]
    return Kitti(root=root)


def test_stereo_object():
    ds = get_dataset()
    inputs = ds.load_frame(0)
    params = ds.load_parameters()
    camera = StereoCamera(**inputs, **params)
    assert camera is not None

def test_stereo_object_shape():
    ds = get_dataset()
    inputs = ds.load_frame(0)
    params = ds.load_parameters()
    camera = StereoCamera(**inputs, **params)
    assert isinstance(camera.left_kp, np.ndarray)
    assert isinstance(camera.right_kp, np.ndarray)
    assert len(camera.left_kp) == len(camera.right_kp)
    assert camera.left_kpoints2d.shape == camera.right_kpoints2d.shape
    assert len(camera.left_kp) == len(camera.left_kpoints2d)
    assert camera.kpoints3d is not None
    assert len(camera.kpoints3d) == camera.left_kpoints2d.shape[0]
    assert camera.kpoints3d.shape[0] == camera.desc3d.shape[0]

