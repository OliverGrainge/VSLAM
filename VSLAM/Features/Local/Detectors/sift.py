from typing import List

import cv2
import numpy as np

from .base import ABCDetector
from ....utils import get_config 

config = get_config()

class SIFT(ABCDetector):
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=config["Stereo"]["MaxFeatures"])

    def detect(self, image: np.ndarray) -> List:
        keypoints = self.sift.detect(image, None)
        return np.array(keypoints)



class SIFTBlocks(ABCDetector):
    def __init__(self, h_blocks=50, w_blocks=45, points_per_block=10):
        self.sift = cv2.SIFT_create(nfeatures=points_per_block)
        self.h_blocks = h_blocks
        self.w_blocks = w_blocks

    def detect(self, image: np.ndarray) -> List:
        H, W = image.shape[:2]
        h_step = H // self.h_blocks
        w_step = W // self.w_blocks
        all_kp = []
        for h_id in range(self.h_blocks):
            for w_id in range(self.w_blocks):
                # Define the current block's slice ranges
                y_start = h_id * h_step
                y_end = (h_id + 1) * h_step
                x_start = w_id * w_step
                x_end = (w_id + 1) * w_step
                # Check to avoid going out of image bounds
                y_end = min(y_end, H)
                x_end = min(x_end, W)
                # Extract the block from the image
                block = image[y_start:y_end, x_start:x_end]
                # Detect keypoints in the block
                keypoints = self.sift.detect(block, None)
                # Adjust keypoints' coordinates to global image coordinates
                for kp in keypoints:
                    kp.pt = (kp.pt[0] + x_start, kp.pt[1] + y_start)
                # Collect all keypoints from all blocks
                all_kp.extend(keypoints)
        return np.array(all_kp)