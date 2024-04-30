import numpy as np
import cv2
from .utils import get_config
from .FeatureTrackers import FeatureTracker
from .Features import LocalFeatures
from .MotionEstimation import MotionEstimation
from .Camera import StereoCamera
from .utils import homogenize
config = get_config()



class VisualSLAM:
    def __init__(self, params: dict):
        self.params = params
        self.feature_extractor = LocalFeatures()
        self.feature_tracker = FeatureTracker()
        self.motion_estimation = MotionEstimation()
        self.cameras = []
    def __call__(self, inputs: dict) -> None:
        if len(self.cameras) == 0:
            cv2.namedWindow("previous")
            cv2.namedWindow("current")
            # create the stereo camera
            cam = StereoCamera(**inputs, **self.params)
            cam.rmat = np.eye(3)
            cam.tvec = np.zeros(3)
            self.cameras.append(cam)
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
                    _trainIdx=idx, _imgIdx=0, _distance=0) for idx in range(len(cam.left_kp))])
            
            sample_matches = np.random.randint(0, len(cam.left_kp), size=(8,))
            matches = matches[sample_matches]
            
            stereo_matches = cv2.drawMatches(
                cam.left_image, 
                cam.left_kp,
                cam.left_image, 
                cam.right_kp, 
                matches,
                None, 
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            cv2.imshow('Stereo Matches', stereo_matches)

            # update the position of the new camera
            cam = StereoCamera(**inputs, **self.params)
            cam.rmat = cam_prev.rmat @ rmat
            cam.tvec = cam_prev.rmat @ tvec + cam_prev.tvec.reshape(-1,1)
            cam.tvec = cam.tvec.flatten()
            cam.x = homogenize(cv2.Rodrigues(cam.rmat)[0], cam.tvec)
            self.cameras.append(cam)

    def trajectory(self):
        traj = np.zeros((len(self.cameras), 3))
        for idx, cam in enumerate(self.cameras):
            traj[idx] = cam.tvec.flatten()
        return traj
