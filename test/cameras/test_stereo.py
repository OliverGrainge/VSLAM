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
