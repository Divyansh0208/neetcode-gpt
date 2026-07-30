import numpy as np
from numpy.typing import NDArray

class Solution:
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z *= -1.0
        np.exp(z, out=z)       
        z += 1.0
        np.divide(1.0, z, out=z)
        
        return np.round(z, 5, out=z)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z[z < 0] = 0
        return z