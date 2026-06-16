import torch
from torch import nn
import torch.nn.functional as F
from typing import *

class MLP(nn.Module):
    def __init__(self, width: List[int]):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(width) - 1):
            self.layers.append(nn.Linear(width[i], width[i + 1]))
            if i < len(width) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)

        return x
    

if __name__ == "__main__":
    model = MLP([64, 32, 10])
    print(model)
