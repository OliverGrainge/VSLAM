from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np


class ABCFeatureMatcher(ABC):
    @abstractmethod
    def match(self, camera):
        pass
