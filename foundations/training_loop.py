import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def gradient(self, X:  NDArray[np.float64], diff: NDArray[np.float64], n: int):
        return (2 / n * (np.dot(X.transpose(), diff)), 2 * np.average(diff))
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        n, m = X.shape
        W, B = (np.zeros(m), 0)
        for epoch_id in range(epochs):
            y_hat = np.dot(X, W) + B
            diff = y_hat - y
            gradient_W, gradient_B = self.gradient(X, diff, X.shape[0])
            W -= lr*gradient_W
            B -= lr*gradient_B
        return (np.round(W, 5), round(B, 5))
