import numpy as np
import cv2
from utils import get_config
from FeatureTrackers import FeatureTracker
from Features import LocalFeatures
from MotionEstimation import MotionEstimation
from Camera import StereoCamera

config = get_config()


class VisualSLAM:
    def __init__(self, params: dict):
        self.params = params
        self.feature_extractor = LocalFeatures
        self.feature_tracker = FeatureTracker()
        self.motion_estimation = MotionEstimation()

        self.cameras = []

    def __call__(self, inputs: dict) -> None:
        if len(self.cameras) == 0:
            # create the stereo camera
            cam = StereoCamera(**inputs, **self.params)
            # find the features in both the left and right iamges
            cam = self.feature_extractor.detectAndCompute(cam)
            # track the features from the left to right image
            cam = self.feature_tracker.track(cam)
            # triangulate those features
            cam.triangulate()
            # add the first camera to the pose-graph
            self.cameras.append(cam)
        else:
            # create the next camera
            cam = StereoCamera(**inputs, **self.params)
            # find the features in both the left and right images
            cam = self.feature_extractor.detectAndCompute(cam)
            # get a copy of the camera will all original detected iamges 
            cam_orig = np.copy(cam)
            # get the previous camera
            cam_prev = self.camera[-1]
            # track the features from the previous left camera image to the current left camera image
            cam_prev, cam = self.feature_tracker.track(cam_prev, cam)
            # estimate the motion between the previous and current camera
            T = self.motion_estimation(cam_prev, cam)
            # update the position of the new camera
            cam_orig.x = T @ cam_prev.x
            # redect all features for t
            # Now track features left to right frame 
            cam = self.feature_tracker.track(cam_orig)
            # Now triangulate point features 
            cam.triangulate()
            # add to the pose graph
            self.cameras.append(cam)
