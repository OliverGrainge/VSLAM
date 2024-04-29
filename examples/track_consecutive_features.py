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
inputs1 = ds.load_frame(0)
inputs2 = ds.load_frame(2)

params = ds.load_parameters()
# create the stereo camera object 
cam1 = StereoCamera(**inputs1, **params)
cam2 = StereoCamera(**inputs2, **params)

# track features in the left and right iamges 
tracking_info = feature_tracker.track(cam1, cam2)

matches = np.array([cv2.DMatch(_queryIdx=idx, 
           _trainIdx=idx, _imgIdx=0, _distance=0) for idx in range(len(tracking_info["points2d_left_cur"]))])

sample_idx = np.random.randint(0, len(matches), size=(8,))



img_matches = cv2.drawMatches(
    cam1.left_image, 
    tracking_info["kp_left_prev"], 
    cam2.left_image, 
    tracking_info["kp_left_cur"], 
    matches[sample_idx],
    None, 
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

# ===================== Tracking Stats ===========================
res = np.linalg.norm(cam1.left_kpoints2d - cam2.left_kpoints2d, axis=1)
print(res.shape)

mean_dist = np.mean(res)
median_dist = np.median(res)
min_dist = np.min(res)
max_dist = np.max(res)
y_var = np.var((cam1.left_kpoints2d - cam2.left_kpoints2d)[:, 1])
x_var = np.var((cam1.left_kpoints2d - cam2.left_kpoints2d)[:, 1])

print("Mean Matched Distance:", mean_dist)
print("Median Matched Distance:", median_dist)
print("Max Matched Distance:", max_dist)
print("Min Matched Distance:", min_dist)
print("x axis variance: ", x_var)
print("y axis variance", y_var)


plt.figure()
res = cam1.left_kpoints2d[:, 0] - cam2.left_kpoints2d[:, 0]
bins = np.linspace(np.min(res), np.max(res), 100)
plt.hist(res, bins=bins)
plt.title("Distirbution of distance between matched points x direction")
plt.xlabel("Distance (pixels)")
plt.ylabel("P(distance)")



plt.figure()
res = cam1.left_kpoints2d[:, 1] - cam2.left_kpoints2d[:, 1]
bins = np.linspace(np.min(res), np.max(res), 100)
plt.hist(res, bins=bins)
plt.title("Distirbution of distance between matched points y direction")
plt.xlabel("Distance (pixels)")
plt.ylabel("P(distance)")
plt.show()


"""