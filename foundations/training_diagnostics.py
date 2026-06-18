import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        with torch.no_grad():
            for layer in model.children():
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    layer_mean = torch.mean(x).item()
                    layer_std = torch.std(x).item()
                    dead_neurons = (x <= 0).all(dim=0)
                    layer_dead_fraction = dead_neurons.float().mean().item()
                    stats.append({
                        'mean': round(layer_mean, 4),
                        'std': round(layer_std, 4),
                        'dead_fraction': round(layer_dead_fraction, 4)
                    })
        return stats
                    

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        stats = []
        model.zero_grad()
        criterion = nn.MSELoss()
        predictions = model(x)
        loss = criterion(predictions, y)
        loss.backward()
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                grad_tensor = layer.weight.grad
                layer_mean = torch.mean(grad_tensor).item()
                layer_std = torch.std(grad_tensor).item()
                layer_norm = torch.norm(grad_tensor).item()
                
                stats.append({
                    'mean': round(layer_mean, 4),
                    'std': round(layer_std, 4),
                    'norm': round(layer_norm, 4)
                })
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for i in range(len(activation_stats)):
            if activation_stats[i]['dead_fraction'] > 0.5:
                return 'dead_neurons'
            if gradient_stats[i]['norm'] > 1000:
                return 'exploding_gradients'
            if gradient_stats[i]['norm'] < 1e-5:
                return 'vanishing_gradients'
        return 'healthy'