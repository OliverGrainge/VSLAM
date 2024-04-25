from abc import ABC, abstractmethod
from typing import List

import numpy as np


class ABCDetector(ABC):
    @abstractmethod
    def detect(image: np.ndarray) -> List:
        pass
