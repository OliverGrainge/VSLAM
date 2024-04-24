from abc import ABC, abstractmethod
from typing import List
import numpy as np

class ABCCamera(ABC):

    @abstractmethod
    def project(self, points: np.ndarray):
        """
        projects 3d (N,3) points to 2d observations (N,2) and returns them
        """

    @property
    @abstractmethod
    def pl(self):
        """
        Returns the left projection matrix of the camera
        """

    @property
    @abstractmethod
    def pr(self):
        """
        Returns the right projection matrix of the camera
        """

    @property
    @abstractmethod
    def distortion(self):
        """
        Returns the distortion parameters of the camera
        """
        
    @property
    @abstractmethod
    def x(self):
        """
        Returns the extrinsic parameters of the camera
        """

    @property
    @abstractmethod
    def k(self):
        """
        Returns the intrinsic calibration matrix of the camera
        """

    @property
    @abstractmethod
    def kp(self):
        """ 
        Returns a list of cv2 keypoints detecting in the image
        """
        pass

    @property
    @abstractmethod
    def kpoints2d(self):
        """ 
        Returns a N,2 np.ndarray of the keypoint locations
        """
        pass

    @property
    @abstractmethod
    def desc2d(self):
        """ 
        Returns a N,D np.ndarray of the keypoint descriptions
        """
        pass

    @property
    @abstractmethod
    def kpoints3d(self):
        """ 
        Returns a N,3 array of the 3d point locations
        """
        pass

    @property
    @abstractmethod
    def desc3d(self):
        """ 
        Returns a N,D array of the 3d point descriptors
        """
        pass


    @property
    @abstractmethod
    def pl(self):
        """
        Sets the left projection matrix of the camera
        """

    @property
    @abstractmethod
    def pr(self):
        """
        Sets the right projection matrix of the camera
        """


    @distortion.setter
    @abstractmethod
    def distortion(self):
        """
        Returns the distortion parameters of the camera
        """

    @x.setter
    @abstractmethod
    def x(self):
        """
        Returns the distortion parameters of the camera
        """

    @k.setter
    @abstractmethod
    def k(self, k: np.ndarray):
        """
        Sets the intrinsic calibration matrix of the camera
        """

    @kp.setter
    @abstractmethod
    def kp(self, kp: List):
        """ 
        Sets the image keypoints with a list of cv2 Keypoints
        """
        pass


    @kpoints2d.setter
    @abstractmethod
    def kpoints2d(self, points: np.ndarray):
        """
        Sets the image with cv2 points2d
        """
        pass

    @desc2d.setter
    @abstractmethod
    def desc2d(self, desc: np.ndarray):
        """
        Sets the descriptors of the 2d keypoints
        """
        pass

    @kpoints3d.setter
    @abstractmethod
    def kpoints3d(self, points: np.ndarray):
        """
        Sets the locations of the 3d points
        """
        pass

    @desc3d.setter
    @abstractmethod
    def desc3d(self, desc: np.ndarray):
        """
        Sets the descriptors of the 3d points
        """
        pass


    
