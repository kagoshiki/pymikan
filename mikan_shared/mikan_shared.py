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
        self.embedding = nn.Embedding(output_dim * input_dim, embedding_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        # (batch_size, input_dim) -> (batch_size, output_dim * input_dim)
        x = x.repeat(1, self.output_dim)

        # (batch_size, output_dim * input_dim) -> (batch_size, output_dim * input_dim, 1)
        x = x.unsqueeze(-1)

        # concat with embedding
        # (batch_size, output_dim * input_dim, 1) -> (batch_size, output_dim * input_dim, 1 + embedding_dim)
        x = torch.cat([x, self.embedding.weight.expand(batch_size, -1, -1)], dim=-1)

        # (batch_size, output_dim * input_dim, 1 + embedding_dim) -> (batch_size * output_dim * input_dim, 1 + embedding_dim)
        x = x.view(-1, 1 + self.embedding.embedding_dim)

        # Edge MLP
        # (batch_size * output_dim * input_dim, 1 + embedding_dim) -> (batch_size * output_dim * input_dim, 1)
        x = self.mlp(x)

        # (batch_size * output_dim * input_dim, 1) -> (batch_size, output_dim, input_dim)
        x = x.view(batch_size, self.output_dim, self.input_dim)

        # (batch_size, output_dim, input_dim) -> (batch_size, output_dim)
        x = x.sum(dim=-1)

        return x



class SharedMIKAN(nn.Module):
    def __init__(
            self,
            widths: List[int],
            edge_mlp_hidden_width: List[int],
            embedding_dim: int = 16,
            activation: Callable = F.relu
        ):
        super().__init__()

        layers = [
            SharedMIKANLayer(
                input_dim=widths[i],
                output_dim=widths[i + 1],
                edge_mlp_hidden_width=edge_mlp_hidden_width,
                embedding_dim=embedding_dim,
                activation=activation
            )
            for i in range(len(widths) - 1)
        ]

        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
