import torch
import torch.nn
from torchtyping import TensorType

class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        M,N = to_reshape.shape
        return torch.reshape(to_reshape, (M*N //2, 2))

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        return torch.mean(to_avg, dim=0)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        return torch.cat((cat_one, cat_two), dim=1)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        return torch.nn.functional.mse_loss(prediction, target)
