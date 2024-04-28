from glob import glob
from os.path import join
from typing import List, Tuple

import numpy as np
import yaml
from PIL import Image

from VSLAM.utils import get_config

from .base import ABCDataset

config = get_config()


def load_poses(pth: str) -> List[np.ndarray]:
    poses = []
    with open(pth, "r") as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=np.float64, sep=" ")
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return np.array(poses)


def load_calib(pth: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = []
    with open(pth, "r") as f:
        for line in f:
            line = line.split(":", 1)[1]
            numbers = [float(number) for number in line.split()]
            data.append(numbers)

    P_l = np.array(data[0]).reshape(3, 4)
    P_r = np.array(data[1]).reshape(3, 4)
    K_l = P_l[:3, :3]
    K_r = P_r[:3, :3]
    return K_l, P_l, K_r, P_r


class Kitti(ABCDataset):
    def __init__(self, sequence="00", root=config["DatasetsDirectory"]):
        self.data_dir = root
        self.sequence_dir = join(
            self.data_dir, "kitti/data_odometry_gray/dataset/sequences/"
        )
        self.image_paths_left = sorted(
            glob(self.sequence_dir + sequence + "/image_0/*.png")
        )
        self.image_paths_right = sorted(
            glob(self.sequence_dir + sequence + "/image_1/*.png")
        )
        self.K_l, self.P_l, self.K_r, self.P_r = load_calib(
            self.sequence_dir + sequence + "/calib.txt"
        )
        self.poses = load_poses(self.sequence_dir + "poses/" + sequence + ".txt")

    def load_frame(self, idx: int) -> dict:
        inputs = {}
        left_image = Image.open(self.image_paths_left[idx]).convert("L")
        right_image = Image.open(self.image_paths_right[idx]).convert("L")
        left_image = np.array(left_image).astype(np.uint8)
        right_image = np.array(right_image).astype(np.uint8)
        inputs["left_image"] = left_image
        inputs["right_image"] = right_image
        return inputs

    def load_parameters(
        self,
    ) -> dict:
        parameters = {}
        parameters["x"] = self.poses[0]
        parameters["kl"] = self.K_l
        parameters["kr"] = self.K_r
        parameters["pl"] = self.P_l
        parameters["pr"] = self.P_r
        parameters["dist"] = np.zeros(5)
        return parameters

    def ground_truth(self):
        gt = [pt[:3, 3] for pt in self.poses]
        return np.array(gt)

    def __len__(
        self,
    ) -> int:
        return len(self.poses)
