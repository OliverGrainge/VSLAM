from abc import ABC, abstractmethod
from typing import List, Union, Tuple
import numpy as np


class ABCLocalFeature(ABC):

    @abstractmethod 
    def detect(image: np.ndarray) -> List:
        pass 

    @abstractmethod
    def compute(image: np.ndarray, keypoints: Union[np.ndarray, List]) -> np.array:
        pass 

    @abstractmethod
    def detectAndCompute(image: np.ndarray) -> Tuple[List, np.ndarray]:
        pass