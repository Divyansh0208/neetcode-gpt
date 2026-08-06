import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        PE = np.zeros((seq_len, d_model))
        pos = np.arange(0, seq_len)[:, np.newaxis]
        i = np.arange(0, d_model, 2)
        agle = pos / 10000**(i / d_model)
        PE[:, 0::2] = np.sin(agle)
        PE[:, 1::2] = np.cos(agle)

        return np.round(PE, 5)
