import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        combined = positive + negative
        dictionary = sorted({word for sentence in combined for word in sentence.split()})

        word_map = {dictionary[i]: float(i + 1) for i, word in enumerate(dictionary)}
        
        tensors = [torch.tensor([word_map[word] for word in sentence.split()]) for sentence in combined]

        return nn.utils.rnn.pad_sequence(tensors, batch_first=True)
