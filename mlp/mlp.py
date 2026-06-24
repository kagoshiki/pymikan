import torch
from torch import nn
import torch.nn.functional as F
from typing import *

class MLP(nn.Module):
    def __init__(self, width: List[int], hidden_activation: Callable = F.relu):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(width) - 1):
            self.layers.append(nn.Linear(width[i], width[i + 1]))
        self.hidden_activation = hidden_activation

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.hidden_activation(x)
        
        return x
    

if __name__ == "__main__":
    model = MLP([64, 32, 10])
    print(model)
