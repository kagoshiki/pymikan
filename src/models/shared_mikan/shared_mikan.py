import torch
from torch import nn
import torch.nn.functional as F

from typing import *


class MLP(nn.Module):
    def __init__(
            self,
            widths: List[int],
            activation: Callable
        ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.activation = activation

        for i in range(len(widths) - 1):
            self.layers.append(nn.Linear(widths[i], widths[i + 1]))
        
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if self.activation == F.relu:
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            else:
                nn.init.xavier_normal_(layer.weight)
            
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        
        return x


class SharedMIKANEdgeWiseEmbLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            shared_edge_mlp_hidden_widths: List[int],
            embedding_dim: int,
            embedding_init_std: float,
            activation: Callable,
            use_layernorm: bool
        ):
        super().__init__()

        self.shared_edge_mlp = MLP([1 + embedding_dim] + shared_edge_mlp_hidden_widths + [1], activation)
        self.embedding = nn.Embedding(output_dim * input_dim, embedding_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_init_std = embedding_init_std

        if use_layernorm:
            self.layernorm = nn.LayerNorm(output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.shared_edge_mlp.reset_parameters()
        nn.init.normal_(self.embedding.weight, std=self.embedding_init_std)

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

        # Shared Edge MLP
        # (batch_size * output_dim * input_dim, 1 + embedding_dim) -> (batch_size * output_dim * input_dim, 1)
        x = self.shared_edge_mlp(x)

        # (batch_size * output_dim * input_dim, 1) -> (batch_size, output_dim, input_dim)
        x = x.view(batch_size, self.output_dim, self.input_dim)

        # (batch_size, output_dim, input_dim) -> (batch_size, output_dim)
        x = x.sum(dim=-1)

        if hasattr(self, 'layernorm'):
            x = self.layernorm(x)

        return x


class SharedMIKANEdgeWiseEmb(nn.Module):
    def __init__(
            self,
            widths: List[int],
            shared_edge_mlp_hidden_widths: List[int],
            embedding_dim: int = 16,
            embedding_init_std: float = 1.0,
            activation: Callable = F.relu,
            use_layernorm: bool = False
        ):
        super().__init__()

        layers = [
            SharedMIKANEdgeWiseEmbLayer(
                input_dim=widths[i],
                output_dim=widths[i + 1],
                shared_edge_mlp_hidden_widths=shared_edge_mlp_hidden_widths,
                embedding_dim=embedding_dim,
                embedding_init_std=embedding_init_std,
                activation=activation,
                use_layernorm=use_layernorm
            )
            for i in range(len(widths) - 1)
        ]

        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SharedMIKANNodeWiseEmbLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            shared_edge_mlp_hidden_widths: List[int],
            in_embedding_dim: int,
            out_embedding_dim: int,
            emb_mixing_mlp_hidden_output_widths: Optional[List[int]],
            embedding_init_std: float,
            activation: Callable,
            use_layernorm: bool
        ):
        super().__init__()

        embedding_dim = in_embedding_dim + out_embedding_dim
        self.shared_edge_mlp = MLP([1 + embedding_dim] + shared_edge_mlp_hidden_widths + [1], activation)
        if emb_mixing_mlp_hidden_output_widths is not None:
            self.emb_mixing_mlp = MLP([embedding_dim] + emb_mixing_mlp_hidden_output_widths, activation)

        self.in_embedding = nn.Embedding(input_dim, in_embedding_dim)
        self.out_embedding = nn.Embedding(output_dim, out_embedding_dim)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_embedding_dim = in_embedding_dim
        self.out_embedding_dim = out_embedding_dim
        self.embedding_init_std = embedding_init_std

        if use_layernorm:
            self.layernorm = nn.LayerNorm(output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.shared_edge_mlp.reset_parameters()
        nn.init.normal_(self.in_embedding.weight, std=self.embedding_init_std)
        nn.init.normal_(self.out_embedding.weight, std=self.embedding_init_std)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        # (batch_size, input_dim) -> (batch_size, output_dim * input_dim)
        x = x.repeat(1, self.output_dim)

        # (batch_size, output_dim * input_dim) -> (batch_size, output_dim * input_dim, 1)
        x = x.unsqueeze(-1)

        # Make edge embeddings
        in_vec = self.in_embedding.weight.repeat(self.output_dim, 1)
        out_vec = self.out_embedding.weight.repeat_interleave(self.input_dim, dim=0)
        edge_embeddings = torch.cat([in_vec, out_vec], dim=-1)

        # Apply Embedding Mixing MLP if it exists
        if hasattr(self, 'emb_mixing_mlp'):
            edge_embeddings = self.emb_mixing_mlp(edge_embeddings)

        edge_embeddings = edge_embeddings.expand(batch_size, -1, -1)

        # concat with embedding
        # (batch_size, output_dim * input_dim, 1) -> (batch_size, output_dim * input_dim, 1 + embedding_dim)
        x = torch.cat([x, edge_embeddings], dim=-1)

        # (batch_size, output_dim * input_dim, 1 + embedding_dim) -> (batch_size * output_dim * input_dim, 1 + embedding_dim)
        x = x.view(-1, 1 + self.in_embedding_dim + self.out_embedding_dim)

        # Shared Edge MLP
        # (batch_size * output_dim * input_dim, 1 + embedding_dim) -> (batch_size * output_dim * input_dim, 1)
        x = self.shared_edge_mlp(x)

        # (batch_size * output_dim * input_dim, 1) -> (batch_size, output_dim, input_dim)
        x = x.view(batch_size, self.output_dim, self.input_dim)

        # (batch_size, output_dim, input_dim) -> (batch_size, output_dim)
        x = x.sum(dim=-1)

        if hasattr(self, 'layernorm'):
            x = self.layernorm(x)

        return x


class SharedMIKANNodeWiseEmb(nn.Module):
    def __init__(
            self,
            widths: List[int],
            shared_edge_mlp_hidden_widths: List[int],
            in_embedding_dim: int = 8,
            out_embedding_dim: int = 8,
            emb_mixing_mlp_hidden_output_widths: Optional[List[int]] = None,
            embedding_init_std: float = 1.0,
            activation: Callable = F.relu,
            use_layernorm: bool = False
        ):
        super().__init__()

        layers = [
            SharedMIKANNodeWiseEmbLayer(
                input_dim=widths[i],
                output_dim=widths[i + 1],
                shared_edge_mlp_hidden_widths=shared_edge_mlp_hidden_widths,
                in_embedding_dim=in_embedding_dim,
                out_embedding_dim=out_embedding_dim,
                emb_mixing_mlp_hidden_output_widths=emb_mixing_mlp_hidden_output_widths,
                embedding_init_std=embedding_init_std,
                activation=activation,
                use_layernorm=use_layernorm
            )
            for i in range(len(widths) - 1)
        ]

        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

