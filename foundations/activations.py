import numpy as np
from numpy.typing import NDArray

class Solution:
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        np.negative(z, out=z)        
        np.exp(z, out=z)             
        np.add(1, z, out=z)          
        np.divide(1, z, out=z)      
        
        return np.round(z, 5, out=z)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(0, z, out=z)