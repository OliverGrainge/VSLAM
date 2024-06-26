from typing import Tuple

import cv2
import numpy as np
import yaml


def homogenize(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transformation = np.eye(4)
    R, _ = cv2.Rodrigues(rvec)
    transformation[:3, :3] = R.squeeze()
    transformation[:3, 3] = tvec.squeeze()
    return transformation


def unhomogenize(T: np.ndarray) -> Tuple[np.ndarray]:
    assert T.shape[0] == 4
    assert T.shape[1] == 4
    rot = T[:3, :3]
    rvec, _ = cv2.Rodrigues(rot)
    tvec = T[:3, 3]
    return rvec.flatten(), tvec


def get_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


def pts2kp(pts: np.ndarray, size=1):
    kp = np.array(
        [
            cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=size)
            for pt in pts.squeeze()
        ]
    )
    return kp


def transform_points3d(points3d: np.ndarray, x: np.ndarray):
    points_homogeneous = np.hstack((points3d, np.ones((points3d.shape[0], 1))))
    points_world_homogeneous = np.dot(x, points_homogeneous.T).T
    points_world = points_world_homogeneous[:, :3]
    return points_world
