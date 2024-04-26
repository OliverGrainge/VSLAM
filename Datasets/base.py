from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np


class ABCDataset(ABC):
    @abstractmethod
    def load_frame(self, idx: int) -> dict:
        pass

    @abstractmethod
    def load_parameters(
        self,
    ) -> dict:
        pass

    @abstractmethod
    def __len__(
        self,
    ) -> int:
        pass

    @abstractmethod
    def ground_truth(
        self,
    ) -> List[np.ndarray]:
        pass
