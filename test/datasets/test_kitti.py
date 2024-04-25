import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import List, Tuple

import numpy as np
import pytest

from Datasets import Kitti


def test_load():
    root = os.path.join(os.getcwd(), "/test/data/")[1:]
    print(root)
    ds = Kitti(root=root)
    assert ds is not None


def get_dataset():
    root = os.path.join(os.getcwd(), "/test/data/")[1:]
    return Kitti(root=root)


def test_length():
    ds = get_dataset()
    assert isinstance(len(ds), int)
    assert len(ds) > 2


def test_inputs():
    ds = get_dataset()
    inputs = ds.load_frame(0)
    assert isinstance(inputs, dict)


def test_parameters_type():
    ds = get_dataset()
    parameters = ds.load_parameters()
    assert isinstance(parameters, dict)


def test_poses_type():
    ds = get_dataset()
    gt = ds.ground_truth()
    assert isinstance(gt, List)
    assert isinstance(gt[0], np.ndarray)


def test_poses_type():
    ds = get_dataset()
    gt = ds.ground_truth()
    assert gt[0].shape[0] == 4
    assert gt[0].shape[1] == 4
    assert np.allclose(gt[0][3, :], np.array([0, 0, 0, 1]))


def test_poses_valid_transformation():
    ds = get_dataset()
    gt = ds.ground_truth()
    assert np.linalg.inv(gt) is not None


def test_images_present():
    ds = get_dataset()
    inputs = ds.load_frame(0)
    assert inputs is not None
    assert "left_image" in inputs.keys()
    assert "right_image" in inputs.keys()


def test_images_dtype():
    ds = get_dataset()
    inputs = ds.load_frame(0)
    left_image = inputs["left_image"]
    right_image = inputs["right_image"]
    assert left_image.dtype == np.uint8
    assert right_image.dtype == np.uint8


def test_images_shape():
    ds = get_dataset()
    inputs = ds.load_frame(0)
    left_image = inputs["left_image"]
    right_image = inputs["right_image"]
    assert left_image.ndim == 2
    assert right_image.ndim == 2
    assert left_image.shape == right_image.shape


def test_parameters():
    ds = get_dataset()
    params = ds.load_parameters()
    assert "kl" in params.keys()
    assert "kr" in params.keys()
    assert "x" in params.keys()
    assert "pl" in params.keys()
    assert "pr" in params.keys()
    assert "dist" in params.keys()


def test_intrinsics():
    ds = get_dataset()
    params = ds.load_parameters()
    kl = params["kl"]
    kr = params["kr"]
    assert kl.shape[0] == 3
    assert kl.shape[1] == 3
    assert kr.shape[0] == 3
    assert kr.shape[1] == 3


def test_extrinsics():
    ds = get_dataset()
    params = ds.load_parameters()
    x = params["x"]
    assert x.shape[0] == 4
    assert x.shape[1] == 4
    assert np.allclose(x[3, :], np.array([0, 0, 0, 1]))


def test_projection():
    ds = get_dataset()
    params = ds.load_parameters()
    pl = params["pl"]
    pr = params["pr"]
    assert pl.shape[0] == 3
    assert pl.shape[1] == 4
    assert pr.shape[0] == 3
    assert pr.shape[1] == 4


def test_distortion():
    ds = get_dataset()
    params = ds.load_parameters()
    dist = params["dist"]
    assert dist.ndim == 1
    assert len(dist) == 5
