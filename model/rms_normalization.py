import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x_arr = np.array(x, dtype=float)
        gamma_arr = np.array(gamma, dtype=float)
        root = np.sqrt(np.mean(x_arr ** 2) + eps)
        res = (x_arr / root) * gamma_arr

        return [round(float(v), 4) for v in res]
       
