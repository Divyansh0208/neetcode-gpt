import torch
import math
from typing import List

class Solution:
    
    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        w = torch.randn(fan_out, fan_in) * std
        return [[round(val.item(), 4) for val in row] for row in w]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        w = torch.randn(fan_out, fan_in) * std
        return [[round(val.item(), 4) for val in row] for row in w]

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        
        weights = []
        for i in range(num_layers):
            f_in = input_dim if i == 0 else hidden_dim
            f_out = hidden_dim
            
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (f_in + f_out))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / f_in)
            else: 
                std = 1.0
                
            w = torch.randn(f_out, f_in) * std
            weights.append(w)
            
        x = torch.randn(input_dim)
        
        stds = []
        for w in weights:
            x = torch.matmul(w, x)
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))
            
        return stds