import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from typing import List, Tuple

import numpy as np
import pytest

from Datasets import Kitti
from VSLAM.Camera import StereoCamera
from VSLAM.Features.Local import Detectors
from VSLAM.Features.Local import Describers


def get_dataset():
    root = os.path.join(os.getcwd(), "/test/data/")[1:]
    return Kitti(root=root)


def get_camera():
    ds = get_dataset()
    inputs = ds.load_frame(0)
    params = ds.load_parameters()
    camera = StereoCamera(**inputs, **params)
    return camera 


def test_instantiate():
    des = Describers.SIFT()
    assert des is not None 

def test_desc():
    det = Detectors.SIFT()
    des = Describers.SIFT()
    camera = get_camera()
    kp = det.detect(camera.left_image)
    desc = des.compute(camera.left_image, kp)
    assert desc is not None 

def test_desc():
    det = Detectors.SIFT()
    des = Describers.SIFT()
    camera = get_camera()
    kp = det.detect(camera.left_image)
    desc = des.compute(camera.left_image, kp)
    assert isinstance(desc, np.ndarray)

def test_desc():
    det = Detectors.SIFT()
    des = Describers.SIFT()
    camera = get_camera()
    kp = det.detect(camera.left_image)
    desc = des.compute(camera.left_image, kp)
    assert len(kp) == len(desc)
    assert desc.ndim == 2