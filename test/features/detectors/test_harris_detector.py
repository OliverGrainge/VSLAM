import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from typing import List, Tuple

import numpy as np
import pytest

from Datasets import Kitti
from VSLAM.Camera import StereoCamera
from VSLAM.Features.Local.Detectors import HARRIS


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
    det = HARRIS()
    assert det is not None 

def test_detect():
    det = HARRIS()
    camera = get_camera()
    kp = det.detect(camera.left_image)
    assert kp is not None 

def test_outputs_dtype():
    det = HARRIS()
    camera = get_camera()
    kp = det.detect(camera.left_image)
    assert isinstance(kp, Tuple)



