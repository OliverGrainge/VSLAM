import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import List, Tuple

import numpy as np
import pytest

from Datasets import Kitti
from VSLAM.Camera import StereoCamera
from VSLAM.FeatureTrackers import FeatureTracker
from VSLAM.Features import LocalFeatures


def get_dataset():
    root = os.path.join(os.getcwd(), "/test/data/")[1:]
    return Kitti(root=root)


def test_stereo_object():
    ds = get_dataset()
    inputs = ds.load_frame(0)
    params = ds.load_parameters()
    camera = StereoCamera(**inputs, **params)
    assert camera is not None


def test_triangulation():
    ds = get_dataset()
    inputs = ds.load_frame(0)
    params = ds.load_parameters()
    camera = StereoCamera(**inputs, **params)
    lf = LocalFeatures()
    tracker = FeatureTracker()
    camera = lf.detectAndCompute(camera)
    camera = tracker.track(camera)
    pts3d = camera.triangulate()
    assert isinstance(pts3d, np.ndarray)
    assert pts3d.ndim == 2
    assert pts3d.shape[1] == 3
    assert np.allclose(pts3d, camera.kpoints3d)


def test_projection():
    ds = get_dataset()
    inputs = ds.load_frame(0)
    params = ds.load_parameters()
    camera = StereoCamera(**inputs, **params)
    lf = LocalFeatures()
    tracker = FeatureTracker()
    camera = lf.detectAndCompute(camera)
    camera = tracker.track(camera)
    pts3d = camera.triangulate()
    pts2d = camera.project(pts3d)
    assert isinstance(pts2d, np.ndarray)
    assert pts2d.ndim == 2
    assert pts2d.shape[1] == 2


def test_reprojection_error():
    ds = get_dataset()
    inputs = ds.load_frame(0)
    params = ds.load_parameters()
    camera = StereoCamera(**inputs, **params)
    lf = LocalFeatures()
    tracker = FeatureTracker()
    camera = lf.detectAndCompute(camera)
    camera = tracker.track(camera)
    pts3d = camera.triangulate()
    pts2d = camera.project(pts3d)
    assert isinstance(pts2d, np.ndarray)
    assert pts2d.ndim == 2
    assert pts2d.shape[1] == 2
    assert np.median(np.abs(pts2d - camera.left_kpoints2d)) < 0.2


def test_translated_reprojection_error():
    ds = get_dataset()
    inputs = ds.load_frame(0)
    params = ds.load_parameters()
    params["x"][3, 3] += 10
    camera = StereoCamera(**inputs, **params)
    lf = LocalFeatures()
    tracker = FeatureTracker()
    camera = lf.detectAndCompute(camera)
    camera = tracker.track(camera)
    pts3d = camera.triangulate()
    pts2d = camera.project(pts3d)
    assert isinstance(pts2d, np.ndarray)
    assert pts2d.ndim == 2
    assert pts2d.shape[1] == 2
    assert np.median(np.abs(pts2d - camera.left_kpoints2d)) < 0.2


def test_translated_vs_static_reprojection_error():
    ds = get_dataset()
    inputs = ds.load_frame(0)
    params = ds.load_parameters()

    lf = LocalFeatures()
    tracker = FeatureTracker()
    params["x"] = np.eye(4)
    camera = StereoCamera(**inputs, **params)
    camera = lf.detectAndCompute(camera)
    camera = tracker.track(camera)
    pts3d = camera.triangulate()

    params["x"][3, 3] += 10
    camera = StereoCamera(**inputs, **params)
    camera = lf.detectAndCompute(camera)
    camera = tracker.track(camera)
    pts3d_t = camera.triangulate()

    pts4d_t = np.hstack(
        (pts3d_t, np.ones(pts3d_t.shape[0]).reshape(pts3d_t.shape[0], 1))
    )
    pts4d_rec = np.linalg.inv(params["x"]) @ pts4d_t.T
    pts3d_rec = pts4d_rec[:3, :] / pts4d_rec[3, :]
    pts3d_rec = pts3d_rec.T

    print(pts3d_rec[:3])
    print(pts3d[:3])
    assert not np.allclose(pts3d_t, pts3d)
    assert np.allclose(pts3d_rec, pts3d)
