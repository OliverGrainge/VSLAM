import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pytest
import numpy as np
from VSLAM.Features import LocalFeatures
from VSLAM.FeatureTrackers import BruteForceTracker, KLTTracker
from VSLAM.utils import get_config
from typing import Tuple


# Utility function for generating a synthetic test image
def generate_test_image():
    return np.random.randint(0, 256, (480, 640), dtype=np.uint8)

# Test if the detector and describer are correctly instantiated
def test_matcher():
    mat = BruteForceTracker()
    assert mat is not None 

def test_matching():
    lf = LocalFeatures()
    mat = BruteForceTracker()
    img1 = generate_test_image()
    img2 = generate_test_image()
    kp1, des1 = lf.detectAndCompute(img1)
    kp2, des2 = lf.detectAndCompute(img2)
    kp1, des1, kp2, des2 = mat.track(img1, img2, kp1, des1, kp2, des2)
    assert isinstance(kp1, Tuple)
    assert isinstance(kp2, Tuple)
    assert isinstance(des1, np.ndarray)
    assert isinstance(des2, np.ndarray)


def test_matching():
    lf = LocalFeatures()
    mat = KLTTracker()
    img1 = generate_test_image()
    img2 = generate_test_image()
    kp1, des1 = lf.detectAndCompute(img1)
    kp2, des2 = lf.detectAndCompute(img2)
    kp1, des1, kp2, des2 = mat.track(img1, img2, kp1, des1, kp2, des2)
    assert isinstance(kp1, Tuple)
    assert isinstance(kp2, Tuple)
    assert isinstance(des1, np.ndarray)
    assert des2 is None




