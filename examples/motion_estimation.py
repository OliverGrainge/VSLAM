import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Datasets import Kitti 
from VSLAM.Features import LocalFeatures
from VSLAM.FeatureTrackers import FeatureTracker
from VSLAM.utils import get_config
from VSLAM.Camera import StereoCamera
from VSLAM.MotionEstimation import MotionEstimation
import matplotlib.pyplot as plt
import cv2

config = get_config()

ds = Kitti()
gt = ds.ground_truth()
feature_extractor = LocalFeatures()
feature_tracker = FeatureTracker()
motion_estimation = MotionEstimation()

# collect the camera inputs
inputs1 = ds.load_frame(0)
inputs2 = ds.load_frame(2)

params = ds.load_parameters()
# create the stereo camera object 
cam1 = StereoCamera(**inputs1, **params)
cam2 = StereoCamera(**inputs2, **params)

# track features in the left and right iamges 
tracking_info = feature_tracker.track(cam1, cam2)


T = motion_estimation(tracking_info)
print("Predicted:", T[:3, 3])
print("Ground Truth", gt[2])