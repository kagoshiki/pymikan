import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from mlp import MLP
from fastkan import FastKAN
from fasterkan import FasterKAN
from mikan import MIKAN
from mikan_shared import SharedMIKAN

import wandb
import time
import datetime

from experiments.fitting import train_model, test_model


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

config = {
    # common
    "model": "FasterKAN",
    "batch_size": 4,
    "widths": [2, 4, 1],
    "optimizer": "Adam",
    "learning_rate": 0.05,
    "num_epoch": 5000,
    "use_layernorm": False,

    # MLP
    "hidden_activation": "tanh",

    # FastKAN/FasterKAN
    "num_grids": 10,

    # MIKAN
    "edge_mlp_hidden_d": 4,
    "edge_mlp_activation": "relu",

    # SharedMIKAN
    "edge_mlp_hidden_widths": [4],
    "embedding_dim": 2,
    "embedding_std": 1.0,
}

ACTIVATION = {
    "relu": F.relu,
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
}

MODEL = {
    "MLP": lambda: MLP(config["widths"], hidden_activation=ACTIVATION[config["hidden_activation"]]).to(device),
    "FastKAN": lambda: FastKAN(config["widths"], num_grids=config["num_grids"]).to(device),
    "FasterKAN": lambda: FasterKAN(config["widths"], num_grids=config["num_grids"]).to(device),
    "MIKAN": lambda: MIKAN(config["widths"], edge_mlp_d=config["edge_mlp_hidden_d"], activation=ACTIVATION[config["edge_mlp_activation"]], use_layernorm=config["use_layernorm"]).to(device),
    "SharedMIKAN": lambda: SharedMIKAN(
        config["widths"], edge_mlp_hidden_widths=config["edge_mlp_hidden_widths"], embedding_dim=config["embedding_dim"], embedding_std=config["embedding_std"], activation=ACTIVATION[config["edge_mlp_activation"]]
    ).to(device),
}

OPTIMIZER = {
    "SGD": lambda params: optim.SGD(params, lr=config["learning_rate"]),
    "Adam": lambda params: optim.Adam(params, lr=config["learning_rate"]),
    "AdamW": lambda params: optim.AdamW(params, lr=config["learning_rate"], weight_decay=1e-5),
}


def main():
    # wandb.init(project="pymikan_xor", name=f"{config['model']}_{datetime.datetime.now()}", config=config)

    batch_size = config["batch_size"]
    num_epoch = config["num_epoch"]

    # XOR dataset
    train_dataset = TensorDataset(
        torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32),
        torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32),
        torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True
    )
    test_loader = DataLoader(
        test_dataset,    
        batch_size = batch_size,
        shuffle = False
    )

    model = MODEL[config["model"]]()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = OPTIMIZER[config["optimizer"]](model.parameters())

    timer = 0
    for epoch in range(num_epoch):
        start = time.perf_counter()
        train_loss = train_model(model, train_loader, optimizer, criterion, device, disable_tqdm=True)
        epoch_time = time.perf_counter() - start
        timer += epoch_time

        # test_loss = test_model(model, test_loader, criterion, device, disable_tqdm=True)
    
        # print(f"Epoch {epoch}:")
        # print(f"Loss: {train_loss}")
        # print(f"Test Loss: {test_loss}")
        # print(f"Time: {epoch_time:.4f} seconds\n")

        # wandb.log({
        #     "train_loss": train_loss,
        #     "test_loss": test_loss,
        #     "epoch_time": epoch_time
        # })

    print(f"Time: {timer}")

    print(f"[0, 0] -> {F.sigmoid(model(torch.tensor([[0, 0]], dtype=torch.float32).to(device))).item():.6f}")
    print(f"[0, 1] -> {F.sigmoid(model(torch.tensor([[0, 1]], dtype=torch.float32).to(device))).item():.6f}")
    print(f"[1, 0] -> {F.sigmoid(model(torch.tensor([[1, 0]], dtype=torch.float32).to(device))).item():.6f}")
    print(f"[1, 1] -> {F.sigmoid(model(torch.tensor([[1, 1]], dtype=torch.float32).to(device))).item():.6f}")

    # wandb.finish()


if __name__ == "__main__":
    main()
