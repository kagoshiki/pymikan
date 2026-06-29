import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import math
import os

from mlp import MLP
from fastkan import FastKAN
from fasterkan import FasterKAN
from mikan import MIKAN
from mikan_shared import SharedMIKAN, SharedMIKANSeparable

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_wandb = False

config = {
    # common
    "model": "SharedMIKANSeparable",
    "task": "XOR",  # XOR, 1D, 2D
    "batch_size": 4,
    "widths": [2, 4, 1],
    "optimizer": "Adam",
    "learning_rate": 0.05,
    "num_epoch": 1000,
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

    # SharedMIKANSeparable
    "in_embedding_dim": 2,
    "out_embedding_dim": 2
}

ACTIVATION = {
    "relu": F.relu,
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
}

MODEL = {
    "MLP": lambda: MLP(config["widths"], hidden_activation=ACTIVATION[config["hidden_activation"]]).to(device),
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

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Data Generators
def generate_xor_data():
    x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    return x, y

def generate_1d_data():
    torch.manual_seed(42)
    x_train = torch.empty(1000, 1).uniform_(-1, 1)
    y_train = torch.sin(x_train)
    x_test = torch.empty(200, 1).uniform_(-1, 1)
    y_test = torch.sin(x_test)
    return x_train, y_train, x_test, y_test

def generate_2d_data():
    torch.manual_seed(42)
    x_train = torch.empty(2000, 2).uniform_(-1, 1)
    y_train = torch.sin(math.pi * x_train[:, [0]]) * torch.cos(math.pi * x_train[:, [1]])
    x_test = torch.empty(400, 2).uniform_(-1, 1)
    y_test = torch.sin(math.pi * x_test[:, [0]]) * torch.cos(math.pi * x_test[:, [1]])
    return x_train, y_train, x_test, y_test

def main():
    task_name = config["task"]
    
    if task_name == "XOR":
        x_train, y_train = generate_xor_data()
        x_test, y_test = generate_xor_data()
        criterion = nn.BCEWithLogitsLoss()
    elif task_name == "1D":
        x_train, y_train, x_test, y_test = generate_1d_data()
        criterion = nn.MSELoss()
    elif task_name == "2D":
        x_train, y_train, x_test, y_test = generate_2d_data()
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown task: {task_name}")

    model = MODEL[config["model"]]()
    param_count = count_trainable_parameters(model)
    
    optimizer = OPTIMIZER[config["optimizer"]](model.parameters())
    
    batch_size = config["batch_size"]
    num_epoch = config["num_epoch"]
    
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training {config['model']} on {task_name} (Params: {param_count})...")
    start_time = time.perf_counter()
    final_loss = 0.0
    for epoch in range(num_epoch):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
            
    execution_time = time.perf_counter() - start_time
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_inputs = x_test.to(device)
        test_targets = y_test.to(device)
        test_outputs = model(test_inputs)
        
        if task_name == "XOR":
            test_probs = torch.sigmoid(test_outputs).cpu()
            predictions = (test_probs > 0.5).float()
            accuracy = (predictions == y_test).float().mean().item()
            print(f"XOR Accuracy: {accuracy * 100:.1f}% | Final Loss: {final_loss:.6f} | Time: {execution_time:.4f}s\n")
        else:
            test_mse = criterion(test_outputs, test_targets).item()
            test_rmse = math.sqrt(test_mse)
            print(f"{task_name} Test RMSE: {test_rmse:.6f} | Time: {execution_time:.4f}s\n")

def experiment_loop():
    models = ["MLP", "FastKAN", "FasterKAN", "MIKAN", "SharedMIKAN", "SharedMIKANSeparable"]
    
    tasks = ["XOR", "1D", "2D"]
    for task in tasks:
        print(f"\n=========================================")
        print(f"STARTING TOY DATASET: {task}")
        print("=========================================")
        
        config["task"] = task
        if task == "XOR":
            config["widths"] = [2, 4, 1]
            config["batch_size"] = 4
            config["num_epoch"] = 1000
            config["optimizer"] = "Adam"
            config["learning_rate"] = 0.05
            config["hidden_activation"] = "tanh"
            config["edge_mlp_hidden_widths"] = [4]
            config["embedding_dim"] = 2
            config["in_embedding_dim"] = 2
            config["out_embedding_dim"] = 2
        else:
            config["widths"] = [1 if task == "1D" else 2, 8, 1]
            config["batch_size"] = 64
            config["num_epoch"] = 500
            config["optimizer"] = "AdamW"
            config["learning_rate"] = 0.01
            config["hidden_activation"] = "relu"
            config["edge_mlp_hidden_widths"] = [16]
            config["embedding_dim"] = 8
            config["in_embedding_dim"] = 16
            config["out_embedding_dim"] = 16
            
        for model_name in models:
            config["model"] = model_name
            try:
                main()
            except Exception as e:
                print(f"Failed to run {model_name} on {task}: {e}")

if __name__ == "__main__":
    main()
    # experiment_loop()
