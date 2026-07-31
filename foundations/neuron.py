import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        z = np.dot(x, w) + b
        if activation == "sigmoid":
            ans = 1/(1+np.exp(-z))
        elif activation == "relu":
            ans = max(z,0)
        
        return float(np.round(ans, 5))