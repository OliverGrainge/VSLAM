import numpy as np
import cv2
from ..utils import homogenize
from scipy.optimize import least_squares

def project_points(points3D, projectionMatrix):
    """
    :param points3D (numpy.array) : size (Nx3)
    :param projectionMatrix (numpy.array) : size(3x4) - final projection matrix (K@[R|t])
    
    Returns:
        points2D (numpy.array) : size (Nx2) - projection of 3D points on image plane
    """
    
    points3D = np.hstack((points3D,np.ones(points3D.shape[0]).reshape(-1,1))) #shape:(Nx4)
    points3D = points3D.T #shape:(4xN)
    pts2D_homogeneous = projectionMatrix @ points3D #shape:(3xN)
    pts2D = pts2D_homogeneous[:2, :]/(pts2D_homogeneous[-1,:].reshape(1,-1)) #shape:(2xN)
    pts2D = pts2D.T
                        
    return pts2D

    
def get_minimization(dof, points3D_world, keypointsLeft, keypointsRight, PL, PR):

    """
    Error function to minimise with the optimisation algorithm

    :param dof            (np.array): size(6) parameter to optimize (rotation and translation vector components)
    :param points3D_world (np.array): size(N,3) 3D points in the world coordinate frame
    :param keypointsLeft  (np.array): size(N,2) 2D projection points on the left image
    :param keypointsRight (np.array): size(N,2) 2D projection points on the right image
    :param PL             (np.array): size(3x4) left projection matrix such that x_L = PL * X_w
    :param PR             (np.array): size(3x4) right projection matrix such that x_R = PR * X_w
                                    (where world coordinates are in the frame of the left camera)

    Returns:
        residual          (np.array): size(4N)
    """

    # Obtain number of points
    N = len(points3D_world)

    # Reshape 3D points in world coordiante frame
    points3D_world = points3D_world.T                                        # shape:(3,N)   
    points3D_world = np.vstack((points3D_world, np.ones(N).reshape(1, -1)))  # shape:(4,N)

    # Unwrap the rotation vector and translation vectors 
    # (in camera coordinate system of the left camera)
    r_vec = np.array([dof[0], dof[1], dof[2]]).reshape(-1, 1)                # shape:(3,1)
    r_mat, _ = cv2.Rodrigues(r_vec)                                          # shape:(3,3)
    t_vec = np.array([dof[3], dof[4], dof[5]]).reshape(-1, 1)                # shape":(3,1)

    # Transform the 3D coordinates in the world coordinate 
    # frame to camera coordinate frame of the left camera
    T_mat = np.hstack((r_mat, t_vec))                                        # shape:(3,4)
    pts3D_left = T_mat @ points3D_world                                      # shape:(3,N)

    # Obtain projection in the left and right image
    pts2D_projection_left = project_points(pts3D_left.T, PL)                 # shape:(N,2)
    pts2D_projection_right = project_points(pts3D_left.T, PR)                # shape:(N,2)

    # Obtain reprojection error in the left and right image
    error_left = (keypointsLeft - pts2D_projection_left)**2                  # shape:(N,2)
    error_right = (keypointsRight - pts2D_projection_right)**2               # shape:(N,2)
    residual = np.vstack((error_left, error_right))                          # shape:(2N,2)
    return residual.flatten()                                                # shape:(4N)






class MotionEstimation3D2D:

    def __call__(self, tracking_info: dict):
        _, rvec, tvec, inliers = cv2.solvePnPRansac(tracking_info["kpoints3d_prev"],
                                        tracking_info["kpoints2d_left_cur"],
                                        tracking_info["kl"],
                                        None,
                                        iterationsCount=2000,
                                        reprojectionError=1,
                                        confidence=0.9999,
                                        flags=cv2.SOLVEPNP_P3P)
        

        rvec, tvec = cv2.solvePnPRefineLM(
            tracking_info["kpoints3d_prev"][inliers],
            tracking_info["kpoints2d_left_cur"][inliers],
            tracking_info["kl"],
            tracking_info["dist"],
            rvec, 
            tvec,
        )

        rmat, _ = cv2.Rodrigues(rvec)
        rmat = rmat.T
        tvec = -np.dot(rmat, tvec)
        return rmat, tvec


   