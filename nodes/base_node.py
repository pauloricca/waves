import numpy as np


class BaseNode:
    def render(self, num_samples: int, **kwargs) -> np.ndarray:
        raise NotImplementedError