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
from efficientkan import KAN
from fastkan import FastKAN
from fasterkan import FasterKAN
from mikan import MIKAN
from mikan_shared import SharedMIKAN, SharedMIKANSeparable
from experiments.feynman import get_feynman_dataset

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_wandb = False

config = {
    # common
    "model": "SharedMIKANSeparable",
    "feynman_eq": "I.6.20a",
    "batch_size": 64,
    "widths": [1, 8, 1],
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

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_feynman_data(f, ranges, num_samples, D_in):
    if not isinstance(ranges[0], list) and not isinstance(ranges[0], tuple):
        lows = [ranges[0]] * D_in
        highs = [ranges[1]] * D_in
    else:
        lows = [r[0] for r in ranges]
        highs = [r[1] for r in ranges]
        
    x = torch.zeros(num_samples, D_in)
    for d in range(D_in):
        x[:, d].uniform_(lows[d], highs[d])
        
    y = f(x)
    if y.ndim == 1:
        y = y.unsqueeze(1)
    elif y.ndim == 2 and y.size(1) != 1:
        y = y.view(num_samples, 1)
        
    return x, y

def main():
    eq_name = config["feynman_eq"]
    
    # Load dataset metadata
    symbol, expr, f, ranges = get_feynman_dataset(eq_name)
    D_in = config["widths"][0]
    
    # Generate datasets
    x_train, y_train = generate_feynman_data(f, ranges, 1000, D_in)
    x_test, y_test = generate_feynman_data(f, ranges, 200, D_in)
    
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    
    batch_size = config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = MODEL[config["model"]]()
    param_count = count_trainable_parameters(model)
    
    # RMSE Loss
    criterion = lambda outputs, targets: torch.sqrt(nn.MSELoss()(outputs, targets))
    optimizer = OPTIMIZER[config["optimizer"]](model.parameters())
    
    num_epoch = config["num_epoch"]
    print(f"Training {config['model']} on Feynman {eq_name} (Params: {param_count})...")
    start_time = time.perf_counter()
    for epoch in range(num_epoch):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()
            
    execution_time = time.perf_counter() - start_time
    
    # Evaluate
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            outputs = model(bx)
            loss = criterion(outputs, by)
            total_test_loss += loss.item()
            
    test_rmse = total_test_loss / len(test_loader)
    print(f"Feynman {eq_name} Test RMSE: {test_rmse:.6f} | Time: {execution_time:.4f}s\n")

def experiment_loop():
    equations = {
        "I.6.20a": {"dim": 1, "desc": "Probability distribution function"},
        "I.12.1": {"dim": 2, "desc": "Frictional force formula"},
        "I.15.3t": {"dim": 4, "desc": "Lorentz time dilation"},
        "II.6.15a": {"dim": 6, "desc": "Dipole potential formula"},
        "I.9.18": {"dim": 9, "desc": "Gravitational potential energy"}
    }
    
    models = ["MLP", "KAN", "FastKAN", "FasterKAN", "MIKAN", "SharedMIKAN", "SharedMIKANSeparable"]
    
    for eq_name, eq_info in equations.items():
        D_in = eq_info["dim"]
        print(f"\nEvaluating Equation {eq_name} ({D_in}D: {eq_info['desc']})...")
        
        config["feynman_eq"] = eq_name
        config["widths"] = [D_in, 8, 1]
        
        for model_name in models:
            config["model"] = model_name
            config["use_layernorm"] = (D_in > 1)
            try:
                main()
            except Exception as e:
                print(f"Failed to run {model_name} on {eq_name}: {e}")

if __name__ == "__main__":
    main()
    # experiment_loop()
