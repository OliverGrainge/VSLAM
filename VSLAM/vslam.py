import numpy as np
import cv2
from .utils import get_config
from .FeatureTrackers import FeatureTracker
from .Features import LocalFeatures
from .MotionEstimation import MotionEstimation
from .Camera import StereoCamera
import copy
import matplotlib.pyplot as plt 
config = get_config()



class VisualSLAM:
    def __init__(self, params: dict):
        self.params = params
        self.feature_extractor = LocalFeatures()
        self.feature_tracker = FeatureTracker()
        self.motion_estimation = MotionEstimation()

        self.cameras = []
        self.count = 0 
    def __call__(self, inputs: dict) -> None:
        if len(self.cameras) == 0:
            cv2.namedWindow("previous")
            cv2.namedWindow("current")
            # create the stereo camera
            cam = StereoCamera(**inputs, **self.params)
            cam.rmat = np.eye(3)
            cam.tvec = np.zeros(3)
            #
            self.cameras.append(cam)
            self.count += 1
        else:
            # create the next camera
            cam = StereoCamera(**inputs, **self.params)
            # get the previous camera 
            cam_prev = self.cameras[-1]
            # track the features from the previous left camera image to the current left camera image
            tracking_info = self.feature_tracker.track(cam_prev, cam)
            # estimate the motion between the previous and current camera
            rmat, tvec = self.motion_estimation(tracking_info)
            matches = np.array([cv2.DMatch(_queryIdx=idx, 
                    _trainIdx=idx, _imgIdx=0, _distance=0) for idx in range(len(tracking_info["kpoints2d_left_cur"]))])

            sample_idx = np.random.randint(0, len(matches), size=(8,))

            #print(tracking_info["kp_left_prev"].shape, tracking_info["kp_left_cur"].shape, len(matches))


            img_matches = cv2.drawMatches(
                cam_prev.left_image, 
                tracking_info["kp_left_prev"], 
                cam.left_image, 
                tracking_info["kp_left_cur"], 
                matches[sample_idx],
                None, 
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            cv2.imshow('Tracked Matched', img_matches)


            matches = np.array([cv2.DMatch(_queryIdx=idx, 
                    _trainIdx=idx, _imgIdx=0, _distance=0) for idx in range(len(cam.left_kp))])
            
            sample_matches = np.random.randint(0, len(cam.left_kp), size=(8,))
            matches = matches[sample_matches]
           #print(cam.left_kp.shape, cam.right_kp.shape, np.max(sample_matches) < len(cam.right_kp))
            
            stereo_matches = cv2.drawMatches(
                cam.left_image, 
                cam.left_kp,
                cam.left_image, 
                cam.right_kp, 
                matches,
                None, 
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            
            print("Stereo Inliders", len(cam.left_kp))
            print("Tracked Inliers:", len(tracking_info["kp_left_prev"]))
            print("==========s")
            cv2.imshow('Stereo Matches', stereo_matches)


            #cv2.waitKey(0)
#

            # update the position of the new camera
            cam = StereoCamera(**inputs, **self.params)
            cam.rmat = cam_prev.rmat @ rmat
            cam.tvec = cam_prev.rmat @ tvec + cam_prev.tvec.reshape(-1,1)
            cam.tvec = cam.tvec.flatten()
            #cam.x = T @ cam_prev.x
            # add to the pose graph
            self.cameras.append(cam)
            self.count += 1 
            return tvec

    def trajectory(self):
        traj = np.zeros((len(self.cameras), 3))
        for idx, cam in enumerate(self.cameras):
            traj[idx] = cam.tvec.flatten()
        return traj
