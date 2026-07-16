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
        activation: callable = F.relu,
        use_layernorm: bool = False
    ):
        super().__init__()

        self.edge_mlp_d = edge_mlp_d

        # EdgeMLP weights
        # Layer1: 1 -> num_hidden
        self.w1 = nn.Parameter(torch.empty(output_dim, input_dim, edge_mlp_d))
        self.b1 = nn.Parameter(torch.empty(output_dim, input_dim, edge_mlp_d))

        # Layer2: num_hidden -> 1
        self.w2 = nn.Parameter(torch.empty(output_dim, input_dim, edge_mlp_d))
        self.b2 = nn.Parameter(torch.empty(output_dim, input_dim))
        self.activation = activation
        
        if use_layernorm:
            self.layernorm = nn.LayerNorm(output_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.activation == F.relu:
            gain = nn.init.calculate_gain('relu')
        elif self.activation == F.sigmoid:
            gain = nn.init.calculate_gain('sigmoid')
        elif self.activation == F.tanh:
            gain = nn.init.calculate_gain('tanh')
        else:
            gain = 1.0
        
        std1 = gain * math.sqrt(2.0 / (1 + self.edge_mlp_d))
        nn.init.normal_(self.w1, std=std1)
        nn.init.zeros_(self.b1)

        std2 = gain * math.sqrt(2.0 / (self.edge_mlp_d + 1))
        nn.init.normal_(self.w2, std=std2)
        nn.init.zeros_(self.b2)

        if hasattr(self, 'layernorm'):
            self.layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        # x: (batch, input_dim)

        # Layer 1
        # (batch, input_dim) -> (batch, input_dim, hidden_dim)
        x = torch.einsum('bi,oih->boih', x, self.w1) + self.b1
        x = self.activation(x)

        # Layer 2
        # (batch, input_dim, hidden_dim) -> (batch, output_dim)
        x = torch.einsum('boih,oih->bo', x, self.w2) + self.b2.sum(dim=1)

        if hasattr(self, 'layernorm'):
            x = self.layernorm(x)
        
        return x


class MIKAN(nn.Module):
    def __init__(
        self,
        layer_sizes: List[int],
        edge_mlp_d=4,
        activation: callable = F.relu,
        use_layernorm: bool = False
    ):
        super().__init__()

        layers = [
            MIKANLayer(
                input_dim=layer_sizes[i],
                output_dim=layer_sizes[i+1],
                edge_mlp_d=edge_mlp_d,
                activation=activation,
                use_layernorm=use_layernorm
            )
            for i in range(len(layer_sizes) - 1)
        ]
        
        self.layers = nn.ModuleList(layers)


    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        
        return x
