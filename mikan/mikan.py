import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import *


class MIKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        edge_mlp_d=4,
        activation: callable = F.relu
    ):
        super().__init__()

        init_scale = 0.1

        # EdgeMLP weights
        # Layer1: 1 -> num_hidden
        self.w1 = nn.Parameter(torch.randn(output_dim, input_dim, edge_mlp_d) * init_scale)
        self.b1 = nn.Parameter(torch.zeros(output_dim, input_dim, edge_mlp_d))

        # Layer2: num_hidden -> 1
        self.w2 = nn.Parameter(torch.randn(output_dim, input_dim, edge_mlp_d) * init_scale)
        self.b2 = nn.Parameter(torch.zeros(output_dim, input_dim))
        self.activation = activation


    def forward(self, x: torch.Tensor):
        # x: (batch, input_dim)

        # Layer 1
        # (batch, input_dim) -> (batch, input_dim, hidden_dim)
        x = torch.einsum('bi,oih->boih', x, self.w1) + self.b1
        x = self.activation(x)

        # Layer 2
        # (batch, input_dim, hidden_dim) -> (batch, output_dim)
        x = torch.einsum('boih,oih->bo', x, self.w2) + self.b2.sum(dim=1)
        
        return x


class MIKAN(nn.Module):
    def __init__(
        self,
        layer_sizes: List[int],
        edge_mlp_d=4,
        activation: callable = F.relu
    ):
        super().__init__()

        layers = [
            MIKANLayer(
                input_dim=layer_sizes[i],
                output_dim=layer_sizes[i+1],
                edge_mlp_d=edge_mlp_d,
                activation=activation
            )
            for i in range(len(layer_sizes) - 1)
        ]
        
        self.layers = nn.ModuleList(layers)


    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        
        return x
