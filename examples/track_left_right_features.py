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
inputs = ds.load_frame(1)
params = ds.load_parameters()

# create the stereo camera object 
camera = StereoCamera(**inputs, **params)

# detect the features in the left and right images 
camera = feature_extractor.detectAndCompute(camera)
# track features in the left and right iamges 
camera = feature_tracker.track(camera)


matches = np.array([cv2.DMatch(_queryIdx=idx, 
           _trainIdx=idx, _imgIdx=0, _distance=0) for idx in range(len(camera.left_kpoints2d))])

sample_idx = np.random.randint(0, len(matches), size=(20,))


img_matches = cv2.drawMatches(
    camera.left_image, 
    camera.left_kp, 
    camera.right_image, 
    camera.right_kp, 
    matches[sample_idx],
    None, 
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()



# ===================== Tracking Stats ===========================
res = np.linalg.norm(camera.left_kpoints2d - camera.right_kpoints2d, axis=1)

mean_dist = np.mean(res)
median_dist = np.median(res)
min_dist = np.min(res)
max_dist = np.max(res)
y_var = np.var((camera.left_kpoints2d - camera.right_kpoints2d)[:, 1])
x_var = np.var((camera.left_kpoints2d - camera.right_kpoints2d)[:, 1])

print("Mean Matched Distance:", mean_dist)
print("Median Matched Distance:", median_dist)
print("Max Matched Distance:", max_dist)
print("Min Matched Distance:", min_dist)
print("x axis variance: ", x_var)
print("y axis variance", y_var)

bins = np.linspace(min_dist, max_dist, 100)
plt.hist(res, bins=bins)
plt.title("Distirbution of distance between matched points")
plt.xlabel("Distance (pixels)")
plt.ylabel("P(distance)")
plt.show()

res = np.abs(camera.left_kpoints2d[:, 1] - camera.right_kpoints2d[:, 1])
bins = np.linspace(min(res), max(res), 100)
plt.hist(res, bins=bins)
plt.title("Distirbution of distance between matched points in y direction")
plt.xlabel("Distance (pixels)")
plt.ylabel("P(distance)")
plt.show()

