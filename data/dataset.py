import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        torch.manual_seed(0)
        tokens = raw_dataset.split()
        starts = torch.randint(0,len(tokens)-context_length,(batch_size,))
        X = []
        Y = []

        for i,start in enumerate(starts):
            start = start.item()
            X.append(tokens[start:start+context_length])
            Y.append(tokens[start+1:start+1+context_length])
        return (X,Y)

