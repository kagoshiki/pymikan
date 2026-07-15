import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from mlp import MLP
from efficientkan import KAN
from fastkan import FastKAN
from fasterkan import FasterKAN
from mikan import MIKAN
from mikan_shared import SharedMIKAN, SharedMIKANSeparable, SharedMIKANSeparableMixing


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    widths = [784, 64, 10]
    mlp_widths = [784, 8, 10]
    # widths = [6, 4, 2, 1]

    models = {
        "MLP": MLP(mlp_widths),
        "MLP-S": MLP([784, 8, 10]),
        "MLP-L": MLP([784, 784, 10]),
        "KAN": KAN(widths, grid_size=8),
        "FastKAN": FastKAN(widths, num_grids=12),
        "FasterKAN": FasterKAN(widths, num_grids=12),
        "MIKAN": MIKAN(widths, edge_mlp_d=4),
        "SharedMIKAN": SharedMIKAN(widths, edge_mlp_hidden_widths=[16], embedding_dim=12, embedding_std=1.0),
        "SharedMIKANSeparable": SharedMIKANSeparable(widths, edge_mlp_hidden_widths=[16], in_embedding_dim=6, out_embedding_dim=6),
        "SharedMIKANSeparableMixing": SharedMIKANSeparableMixing(widths, edge_mlp_hidden_widths=[64], in_embedding_dim=6, out_embedding_dim=6, mixed_embedding_dim=12),
    }

    for name, model in models.items():
        print(f"{name}:")
        print(f"  Total parameters: {count_parameters(model)}")
        print(f"  Trainable parameters: {count_trainable_parameters(model)}")
        print()


if __name__ == "__main__":
    main()