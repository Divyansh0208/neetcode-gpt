import numpy as np
from numpy.typing import NDArray
from typing import List

class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        activation = x
        
        num = len(weights)
        
        for i in range(num):
            z = np.dot(activation, weights[i]) + biases[i]
            
            if i < num - 1:
                activation = np.maximum(0.0, z)
            else:
                activation = z
                
        return np.round(activation, 5)