import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        mean = np.mean(x)
        var = (np.std(x)) ** 2
        eps = 1e-5
        x_hat = (x - mean) / math.sqrt(var + eps)
        out = gamma * x_hat + beta
        return np.round(out, 5)
