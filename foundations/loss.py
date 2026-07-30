import numpy as np
from numpy.typing import NDArray

class Solution:
    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        np.clip(y_pred, 1e-7, 1 - 1e-7, out=y_pred)
        loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return round(float(loss), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        np.clip(y_pred, 1e-7, 1 - 1e-7, out=y_pred)
        np.log(y_pred, out=y_pred)
        y_pred *= y_true
        loss = -np.sum(y_pred) / y_true.shape[0]
        
        return round(float(loss), 4)