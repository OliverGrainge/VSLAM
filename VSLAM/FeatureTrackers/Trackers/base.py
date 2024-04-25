from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np


class ABCFeatureTracker(ABC):
    @abstractmethod
    def track(self, camera1, camera2=None) -> Tuple:
        pass
