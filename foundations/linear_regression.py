import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        predictions = X @ weights
        return np.round(predictions, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        model_prediction -= ground_truth
        np.square(model_prediction, out=model_prediction)
        mse = np.mean(model_prediction)
        
        return round(float(mse), 5)