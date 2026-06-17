import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from mlp import MLP
from fastkan import FastKAN
from fasterkan import FasterKAN
from mikan import MIKAN
from mikan_shared import SharedMIKAN

import wandb
from tqdm import tqdm
import time
import datetime

from fitting import train_model, test_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

config = {
    # common
    "model": "SharedMIKAN",
    "batch_size": 64,
    "widths": [784, 128, 10],
    "optimizer": "AdamW",
    "learning_rate": 0.005,
    "num_epoch": 20,

    # FastKAN/FasterKAN
    "num_grids": 10,

    # MIKAN
    "edge_mlp_hidden_d": 4,
    "edge_mlp_activation": "relu",

    # SharedMIKAN
    "edge_mlp_hidden_widths": [32],
    "embedding_dim": 8,
}

ACTIVATION = {
    "relu": F.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
}

MODEL = {
    "MLP": lambda: MLP(config["widths"]).to(device),
    "FastKAN": lambda: FastKAN(config["widths"], num_grids=config["num_grids"]).to(device),
    "FasterKAN": lambda: FasterKAN(config["widths"], num_grids=config["num_grids"]).to(device),
    "MIKAN": lambda: MIKAN(config["widths"], edge_mlp_d=config["edge_mlp_hidden_d"], activation=ACTIVATION[config["edge_mlp_activation"]]).to(device),
    "SharedMIKAN": lambda: SharedMIKAN(
        config["widths"], edge_mlp_hidden_widths=config["edge_mlp_hidden_widths"], embedding_dim=config["embedding_dim"], activation=ACTIVATION[config["edge_mlp_activation"]]
    ).to(device),
}

OPTIMIZER = {
    "SGD": lambda params: optim.SGD(params, lr=config["learning_rate"]),
    "AdamW": lambda params: optim.AdamW(params, lr=config["learning_rate"], weight_decay=1e-5),
}


def main():
    wandb.init(project="pymikan", name=f"{config['model']}_{datetime.datetime.now()}", config=config)

    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    num_epoch = config["num_epoch"]

    train_dataset = datasets.MNIST(
        './data',
        train = True,
        download = True,
        transform = ToTensor()
    )
    test_dataset = datasets.MNIST(
        './data',
        train = False,
        download=True,
        transform = ToTensor()
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

    criterion = nn.CrossEntropyLoss()

    optimizer = OPTIMIZER[config["optimizer"]](model.parameters())

    timer = 0
    for epoch in range(num_epoch):
        start = time.perf_counter()
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device, True)
        epoch_time = time.perf_counter() - start
        timer += epoch_time

        test_loss, test_acc = test_model(model, test_loader, criterion, device, True)
    
        print(f"Epoch {epoch}:")
        print(f"Train Accuracy: {train_acc}")
        print(f"Test Accuracy: {test_acc}")
        print(f"Time: {epoch_time:.4f} seconds\n")

        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_time": epoch_time
        })

    print(f"Time: {timer}")

    wandb.finish()


if __name__ == "__main__":
    main()
