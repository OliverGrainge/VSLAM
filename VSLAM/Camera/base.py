from abc import ABC, abstractmethod
from typing import List

import numpy as np


class ABCCamera(ABC):

    @abstractmethod
    def project(self, points: np.ndarray):
        """
        projects 3d (N,3) points to 2d observations (N,2) and returns them
        """



    
