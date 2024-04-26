import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Datasets import Kitti 
from VSLAM.Features import LocalFeatures
from VSLAM.FeatureTrackers import FeatureTracker
from VSLAM.utils import get_config
from VSLAM.Camera import StereoCamera
import matplotlib.pyplot as plt
import cv2

config = get_config()

ds = Kitti()
feature_extractor = LocalFeatures()
feature_tracker = FeatureTracker()

# collect the camera inputs
inputs1 = ds.load_frame(1)

params = ds.load_parameters()

# create the stereo camera object 
cam1 = StereoCamera(**inputs1, **params)


# detect the features in the left and right images 
cam1 = feature_extractor.detectAndCompute(cam1)
# track features in the left and right iamges 
cam1 = feature_tracker.track(cam1)
cam1.triangulate()



# ===================== Tracking Stats ===========================
depths = cam1.kpoints3d[:, 2]
mean_dist = np.mean(depths)
median_dist = np.median(depths)
min_dist = np.min(depths)
max_dist = np.max(depths)
var = np.var(depths)

print("Mean Depths:", mean_dist)
print("Median Depths:", median_dist)
print("Max Depths:", max_dist)
print("Min Depths:", min_dist)
print("Depths variance: ", var)


bins = np.linspace(min_dist, max_dist, 100)
plt.hist(depths, bins=bins)
plt.title("Distirbution of triangulated depths")
plt.xlabel("Depth (M)")
plt.ylabel("P(depth)")
plt.show()


