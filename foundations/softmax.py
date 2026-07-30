import numpy as np
from numpy.typing import NDArray

class Solution:
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z -= np.max(z)
        np.exp(z, out=z)
        z /= np.sum(z)
        return np.round(z, 4, out=z)