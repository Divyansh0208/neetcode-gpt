import torch
import torch.nn as nn
from typing import List, Dict

class Solution:
    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        stats = []
        with torch.no_grad():
            for layer in model:
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    mean = torch.mean(x).item()
                    std = torch.std(x).item()
                    dead = (x<=0).all(dim=0)
                    dead_fraction = dead.float().mean().item()
                    stats.append({
                        "mean": round(mean, 4),
                        "std": round(std, 4),
                        "dead_fraction": round(dead_fraction, 4)
                    })
        return stats
    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad()
        y_hat = model(x)
        MSE = nn.MSELoss()
        loss = MSE(y_hat, y)
        loss.backward()
        stats = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                mean = torch.mean(grad).item()
                std = torch.std(grad).item()
                norm = torch.norm(grad).item()
                stats.append({
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "norm": round(norm, 4)
                })
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        for stats in activation_stats:
            if stats["dead_fraction"] > 0.5:
                return "dead_neurons"
        
        for stats in gradient_stats:
            if stats["norm"] > 1000:
                return "exploding_gradients"
            
            elif stats["norm"] < 1e-5:
                return "vanishing_gradients"
        
        for stats in activation_stats:
            if stats["std"] < 0.1:
                return "vanishing_gradients"
            
            elif stats["std"] > 10:
                return "exploding_gradients"
        
        return "healthy"
