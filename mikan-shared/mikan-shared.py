import torch
from torch import nn
import torch.nn.functional as F

from typing import *


class SharedEdgeMLP(nn.Module):
    def __init__(
            self,
            width: List[int],
            activation: Callable
        ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        for i in range(len(width) - 1):
            self.layers.append(nn.Linear(width[i], width[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        
        return x


class SharedMIKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            edge_mlp_hidden_width: List[int],
            embedding_dim: int,
            activation: Callable
        ):
        super().__init__()
        self.mlp = SharedEdgeMLP([1 + embedding_dim] + edge_mlp_hidden_width + [1], activation)
    
    def forward(self, x):
        pass


class SharedMIKAN(nn.Module):
    def __init__(
            self,
            widths: List[int],
            edge_mlp_hidden_width: List[int],
            embedding_dim: int = 16,
            activation: Callable = F.relu
        ):
        super().__init__()
    
    def forward(self, x):
        pass
