import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import Tuple

import numpy as np
import pytest

from VSLAM.Features import LocalFeatures
from VSLAM.utils import get_config


# Utility function for generating a synthetic test image
def generate_test_image():
    return np.random.randint(0, 256, (480, 640), dtype=np.uint8)


# Test if the detector and describer are correctly instantiated
def test_detector_describer_initialization():
    lf = LocalFeatures()
    assert lf.detector is not None
    assert lf.describer is not None


# Test the detect method
def test_detect():
    lf = LocalFeatures()
    image = generate_test_image()
    keypoints = lf.detect(image)
    assert isinstance(
        keypoints, Tuple
    )  # Or any other assertion relevant to your keypoints


# Test the compute method
def test_compute():
    lf = LocalFeatures()
    image = generate_test_image()
    keypoints = lf.detect(image)  # First detect keypoints
    descriptors = lf.compute(image, keypoints)
    assert isinstance(descriptors, np.ndarray)  # Check the type of descriptors


# Test the detectAndCompute method
def test_detect_and_compute():
    lf = LocalFeatures()
    image = generate_test_image()
    keypoints, descriptors = lf.detectAndCompute(image)
    assert isinstance(keypoints, Tuple)
    assert isinstance(descriptors, np.ndarray)
    assert len(keypoints) == descriptors.shape[0]  # Check consistency
