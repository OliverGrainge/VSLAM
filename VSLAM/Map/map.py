from typing import List, Tuple, Dict
from VSLAM.Camera.stereocamera import StereoCamera
from VSLAM.FeatureMatchers import FeatureMatcher
import numpy as np
from ..utils import get_config
from .mapmatcher import MapMatcher
import cv2
from ..utils import homogenize, transform_points3d, pts2kp
from scipy.optimize import least_squares
from ..Backend import data_assocation, reprojection_error

config = get_config()


class Map:
    def __init__(self):
        super().__init__()
        self.cameras = []
        self.window = config["Map"]["WindowSize"]
        self.count = 0

    def __call__(self, camera: StereoCamera, tracking_info):
        self.cameras.append(camera)
        query_cam = self.cameras[-1]
        
        if len(self.cameras) >= self.window: 
            da = data_assocation(self.cameras, self.window)

            
            pts3d = query_cam.kpoints3d
            obs = query_cam.left_kpoints2d
            res_self, _ = reprojection_error(pts3d, np.zeros(3), np.zeros(3), np.zeros(5), query_cam.kl, obs)
            
            

            for assoc, cam in zip(da, self.cameras[-self.window:-1]):
                pts3d = transform_points3d(query_cam.kpoints3d[assoc[0]], query_cam.x)
                obs = cam.left_kpoints2d[assoc[1]]
                x = cam.x 
                x = np.linalg.inv(x)
                rvec = cv2.Rodrigues(x[:3, :3])[0]
                tvec = x[:3, 3]
                res, pp = reprojection_error(pts3d, rvec, tvec, np.zeros(5), cam.kl, obs)
                print(np.mean(res_self), np.mean(res))
                """
                matches = np.array(
                    [
                        cv2.DMatch(_queryIdx=assoc[0][idx], _trainIdx=assoc[1][idx], _imgIdx=0, _distance=0)
                        for idx in range(len(assoc[0]))
                    ]
                )
                sample_matches = np.random.randint(0, len(assoc[0]), size=(8,))
                matches = matches[sample_matches]
                stereo_matches = cv2.drawMatches(
                    query_cam.left_image,
                    query_cam.left_kp,
                    cam.left_image,
                    cam.left_kp,
                    matches,
                    None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
                matches = np.array(
                    [
                        cv2.DMatch(_queryIdx=assoc[0][idx], _trainIdx=idx, _imgIdx=0, _distance=0)
                        for idx in range(len(assoc[0]))
                    ]
                )
                matches = matches[sample_matches]
                stereo_matches2 = cv2.drawMatches(
                    query_cam.left_image,
                    query_cam.left_kp,
                    cam.left_image,
                    pts2kp(pp),
                    matches,
                    None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
                print(np.mean(res))
                print("hi")
                import matplotlib.pyplot as plt 
                plt.hist(res, bins=np.linspace(res.min(), res.max(), 100))
                plt.show()
                cv2.imshow("Map Matches", stereo_matches)
                cv2.imshow("Map Reprojections", stereo_matches2)
                cv2.waitKey(0)
                """


        self.count += 1

    def local_map(self):
        return np.vstack([pt.kpoints3d for pt in self.cameras[-self.window :]])

    def global_map(self):
        return np.vstack([pt.kpoints3d for pt in self.cameras[-self.window :]])

    def local_traj(self):
        return np.vstack([pt.x[:3, 3] for pt in self.cameras[-self.window :]])

    def global_traj(self):
        return np.vstack([pt.x[:3, 3] for pt in self.cameras])
