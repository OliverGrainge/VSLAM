from abc import ABC, abstractmethod
from typing import List, Union, Tuple
import numpy as np


class ABCFeatureTracker(ABC):

    @abstractmethod 
    def track(
        self,
        image_ref: np.ndarray,
        image_cur: np.ndarray,
        kp_ref: List,
        kp_cur: List,
        desc_ref: np.ndarray, 
        desc_cur: np.ndarray,
    ) -> Tuple[List, np.ndarray, List, np.ndarray]:
        pass 
