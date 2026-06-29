import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import datetime

from mlp import MLP
from efficientkan import KAN
from fastkan import FastKAN
from fasterkan import FasterKAN
from mikan import MIKAN
from mikan_shared import SharedMIKAN, SharedMIKANSeparable

import wandb
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

use_wandb = False

config = {
    # common
    "model": "SharedMIKANSeparable",
    "feynman_eq": "I.6.2",
    "batch_size": 64,
    "widths": [2, 2, 1, 1],
    "optimizer": "AdamW",
    "learning_rate": 0.01,
    "num_epoch": 300,
    "use_layernorm": False,

    # MLP
    "hidden_activation": "relu",

    # FastKAN/FasterKAN
    "num_grids": 8,

    # MIKAN
    "edge_mlp_hidden_d": 8,
    "edge_mlp_activation": "tanh",

    # SharedMIKAN / SharedMIKANSeparable
    "edge_mlp_hidden_widths": [16, 16],
    "embedding_dim": 8,
    "embedding_std": 1.0,
    "in_embedding_dim": 4,
    "out_embedding_dim": 4,
}

ACTIVATION = {
    "relu": F.relu,
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
}

MODEL = {
    "MLP": lambda: MLP(config["widths"], hidden_activation=ACTIVATION[config["hidden_activation"]]).to(device),
    "KAN": lambda: KAN(config["widths"]).to(device),
    "FastKAN": lambda: FastKAN(config["widths"], num_grids=config["num_grids"], use_layernorm=config["use_layernorm"]).to(device),
    "FasterKAN": lambda: FasterKAN(config["widths"], num_grids=config["num_grids"]).to(device),
    "MIKAN": lambda: MIKAN(
        config["widths"], 
        edge_mlp_d=config["edge_mlp_hidden_d"], 
        activation=ACTIVATION[config["edge_mlp_activation"]],
        use_layernorm=config["use_layernorm"]
    ).to(device),
    "SharedMIKAN": lambda: SharedMIKAN(
        config["widths"],
        edge_mlp_hidden_widths=config["edge_mlp_hidden_widths"],
        embedding_dim=config["embedding_dim"],
        embedding_std=config["embedding_std"],
        activation=ACTIVATION[config["edge_mlp_activation"]],
        use_layernorm=config["use_layernorm"]
    ).to(device),
    "SharedMIKANSeparable": lambda: SharedMIKANSeparable(
        config["widths"],
        edge_mlp_hidden_widths=config["edge_mlp_hidden_widths"],
        in_embedding_dim=config["in_embedding_dim"],
        out_embedding_dim=config["out_embedding_dim"],
        embedding_std=config["embedding_std"],
        activation=ACTIVATION[config["edge_mlp_activation"]],
        use_layernorm=config["use_layernorm"]
    ).to(device),
}

OPTIMIZER = {
    "SGD": lambda params: optim.SGD(params, lr=config["learning_rate"]),
    "Adam": lambda params: optim.Adam(params, lr=config["learning_rate"]),
    "AdamW": lambda params: optim.AdamW(params, lr=config["learning_rate"], weight_decay=1e-5),
}

# Define Feynman target functions
def f_i_6_2(x):
    theta, sigma = x[:, 0], x[:, 1]
    return torch.exp(- (theta ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * np.pi * sigma ** 2)

def f_i_6_2b(x):
    theta, theta1, sigma = x[:, 0], x[:, 1], x[:, 2]
    return torch.exp(- ((theta - theta1) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * np.pi * sigma ** 2)

def f_i_9_18(x):
    a, b, c, d, e, f = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
    return a / ((b - 1) ** 2 + (c - d) ** 2 + (e - f) ** 2 + 1e-6)

def f_i_12_11(x):
    a, theta = x[:, 0], x[:, 1]
    return 1.0 + a * torch.sin(theta)

def f_i_13_12(x):
    a, b = x[:, 0], x[:, 1]
    return a * (1.0 / (b + 1e-6) - 1.0)

FEYNMAN_FUNCTIONS = {
    "I.6.2": {"func": f_i_6_2, "input_dim": 2},
    "I.6.2b": {"func": f_i_6_2b, "input_dim": 3},
    "I.9.18": {"func": f_i_9_18, "input_dim": 6},
    "I.12.11": {"func": f_i_12_11, "input_dim": 2},
    "I.13.12": {"func": f_i_13_12, "input_dim": 2},
}

def generate_data(eq_name, num_samples):
    info = FEYNMAN_FUNCTIONS[eq_name]
    func = info["func"]
    input_dim = info["input_dim"]

    if eq_name == "I.6.2":
        theta = torch.rand(num_samples, 1) * 6.0 - 3.0
        sigma = torch.rand(num_samples, 1) * 2.5 + 0.5
        x = torch.cat([theta, sigma], dim=1)
    elif eq_name == "I.6.2b":
        theta = torch.rand(num_samples, 1) * 6.0 - 3.0
        theta1 = torch.rand(num_samples, 1) * 2.0 - 1.0
        sigma = torch.rand(num_samples, 1) * 2.5 + 0.5
        x = torch.cat([theta, theta1, sigma], dim=1)
    elif eq_name == "I.9.18":
        a = torch.rand(num_samples, 1) * 4.0 + 1.0
        others = torch.rand(num_samples, 5) * 3.5 + 1.5
        x = torch.cat([a, others], dim=1)
    elif eq_name == "I.12.11":
        a = torch.rand(num_samples, 1) * 5.0
        theta = torch.rand(num_samples, 1) * (2 * np.pi) - np.pi
        x = torch.cat([a, theta], dim=1)
    elif eq_name == "I.13.12":
        a = torch.rand(num_samples, 1) * 5.0
        b = torch.rand(num_samples, 1) * 4.8 + 0.2
        x = torch.cat([a, b], dim=1)
    else:
        x = torch.rand(num_samples, input_dim)

    y = func(x).unsqueeze(1)
    return TensorDataset(x, y)

# Regression fit functions (similar to train_model / test_model in fitting.py but for regression)
def train_model_reg(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test_model_reg(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def main():
    # Adjust widths input_dim automatically to avoid mismatches
    eq_name = config["feynman_eq"]
    input_dim = FEYNMAN_FUNCTIONS[eq_name]["input_dim"]
    config["widths"][0] = input_dim

    if use_wandb:
        wandb.init(project="pymikan_feynman", name=f"{config['model']}_{eq_name}_{datetime.datetime.now()}", config=config)

    batch_size = config["batch_size"]
    num_epoch = config["num_epoch"]

    print(f"Generating data for {eq_name} (Input Dim: {input_dim})...")
    train_dataset = generate_data(eq_name, 2000)
    test_dataset = generate_data(eq_name, 500)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Initializing {config['model']} with widths={config['widths']}...")
    model = MODEL[config["model"]]()
    criterion = lambda outputs, targets: torch.sqrt(nn.MSELoss()(outputs, targets))     # RMSE Loss
    optimizer = OPTIMIZER[config["optimizer"]](model.parameters())

    timer = 0
    best_test_loss = float("inf")

    for epoch in range(num_epoch):
        start = time.perf_counter()
        train_loss = train_model_reg(model, train_loader, optimizer, criterion, device)
        epoch_time = time.perf_counter() - start
        timer += epoch_time

        test_loss = test_model_reg(model, test_loader, criterion, device)
        if test_loss < best_test_loss:
            best_test_loss = test_loss

        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == num_epoch - 1:
            print(f"Epoch {epoch}:")
            print(f"Train Loss (RMSE): {train_loss:.6f}")
            print(f"Test Loss (RMSE): {test_loss:.6f}")
            print(f"Time: {epoch_time:.4f} seconds\n")

        if use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "test_loss": test_loss,
                "epoch_time": epoch_time
            })

    print(f"Training Finished. Best Test RMSE: {best_test_loss:.6f}")
    print(f"Total Time: {timer:.4f} seconds")

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
