import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.models.shared_mikan import SharedMIKANEdgeWiseEmb
from experiments.fitting_reg import train_model, test_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 43

font_size = 20

config = {
    "widths": [2, 1, 1],
    "batch_size": 32,
    "num_epoch": 1000,
    "learning_rate": 0.01,
    "num_samples": 1000,
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# y = exp(sin(pi * x1) + x2^2)
def generate_data(num_samples: int = 1000):
    x = torch.empty(num_samples, 2).uniform_(-1, 1)
    y = torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)

    return x, y


def get_edge_output(x: torch.Tensor, mlp: nn.Module, embedding: torch.Tensor):
    batch_size = x.size(0)
    embedding_expanded = embedding.unsqueeze(0).expand(batch_size, -1)
    input = torch.cat([x, embedding_expanded], dim=1)
    output = mlp(input)

    return output


def plot_edge_functions(model: SharedMIKAN):
    layer1 = model.layers[0]
    layer2 = model.layers[1]

    mlp1 = layer1.mlp
    mlp2 = layer2.mlp

    embedding1 = layer1.embedding.weight[0]
    embedding2 = layer1.embedding.weight[1]
    embedding3 = layer2.embedding.weight[0]

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    titles = ["x1 -> h", "x2 -> h", "h -> y"]

    for i in range(3):
        x = torch.linspace(-1, 1, 100).unsqueeze(1).to(device)
        if i == 0:
            mlp = mlp1
            embedding = embedding1
        elif i == 1:
            mlp = mlp1
            embedding = embedding2
        else:
            mlp = mlp2
            embedding = embedding3

        y = get_edge_output(x, mlp, embedding)
        ax[i].plot(x.cpu().numpy(), y.detach().cpu().numpy(), linewidth=3)
        ax[i].set_title(titles[i], fontsize=font_size)
        ax[i].set_xlabel("x", fontsize=font_size)
        ax[i].set_ylabel("y", fontsize=font_size)
        ax[i].tick_params(axis='both', which='major', labelsize=font_size)
        # ax[i].set_box_aspect(1)

    plt.tight_layout()
    fig.savefig("results/edge_functions.svg", bbox_inches="tight")


def main():
    # Dataset
    x, y = generate_data(num_samples=config["num_samples"])
    print(f"y - Max: {y.max().item():.4f}, Min: {y.min().item():.4f}")
    # Normalize y
    y = (y - y.min()) / (y.max() - y.min())
    print(f"y (normalized) - Max: {y.max().item():.4f}, Min: {y.min().item():.4f}")
    dataset = torch.utils.data.TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Model
    model = SharedMIKAN(widths=config["widths"], edge_mlp_hidden_widths=[16], embedding_dim=8, activation=F.tanh).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()

    # Training loop
    for epoch in tqdm(range(config["num_epoch"]), desc="Training", leave=False):
        avg_loss = train_model(model, data_loader, optimizer, criterion, device, True)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{config['num_epoch']}] | Loss: {avg_loss:.8f}")
        # print(f"Epoch [{epoch + 1}/{config['num_epoch']}], Loss: {avg_loss:.4f}")

    # Plot edge functions
    plot_edge_functions(model)    


if __name__ == "__main__":
    set_seed(seed)
    main()
