from abc import ABC, abstractmethod
from typing import List, Union 
import numpy as np



class ABCDescriber(ABC):

    @abstractmethod
    def compute(image: np.ndarray, keypoints: Union[List, np.ndarray]) -> np.ndarray:
        pass 