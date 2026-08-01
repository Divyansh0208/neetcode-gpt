import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x = np.array(x)
        gamma = np.array(gamma)
        root = np.sqrt(np.mean(x ** 2) + eps)
        x_hat = x / root
        return np.round(gamma * x_hat, 4).tolist()