import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from src.models.mlp import MLP
from src.models.efficientkan import KAN
from src.models.fastkan import FastKAN
from src.models.fasterkan import FasterKAN
from src.models.mikan import MIKAN
from src.models.shared_mikan import SharedMIKANEdgeWiseEmb, SharedMIKANNodeWiseEmb

import wandb
from tqdm import tqdm
import time
import datetime

from src.experiments.fitting_class import train_model, test_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

use_wandb = True

project_name = "mikan-fashion-mnist-embonly-v2"

config = {
    # common
    "model": "KAN",
    "batch_size": 64,
    "widths": [784, 64, 10],
    "optimizer": "AdamW",
    "learning_rate": 0.005,
    "num_epoch": 30,
    "embedding_only_num_epoch": 20,
    "use_layernorm": True,

    # MLP
    "hidden_activation": "relu",

    # FastKAN/FasterKAN
    "num_grids": 12,

    # MIKAN
    "edge_mlp_hidden_d": 4,
    "edge_mlp_activation": "relu",

    # SharedMIKAN
    "shared_edge_mlp_hidden_widths": [16],
    "embedding_init_std": 1.0,

    # SharedMIKAN EdgeWiseEmbedding
    "embedding_dim": 12,

    # SharedMIKAN NodeWiseEmbedding
    "in_embedding_dim": 6,
    "out_embedding_dim": 6,

    # SharedMIKAN NodeWiseEmbeddingWithMixing
    "emb_mixing_mlp_hidden_output_widths": [12, 12]
}

ACTIVATION = {
    "relu": F.relu,
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
}

MODEL = {
    "MLP": lambda: MLP([784, 64, 10], hidden_activation=ACTIVATION[config["hidden_activation"]]).to(device),
    "MLP-S" : lambda: MLP([784, 8, 10], hidden_activation=ACTIVATION[config["hidden_activation"]]).to(device),
    "MLP-L" : lambda: MLP([784, 784, 10], hidden_activation=ACTIVATION[config["hidden_activation"]]).to(device),
    "KAN": lambda: KAN(config["widths"]).to(device),
    "FastKAN": lambda: FastKAN(config["widths"], num_grids=config["num_grids"]).to(device),
    "FasterKAN": lambda: FasterKAN(config["widths"], num_grids=config["num_grids"]).to(device),
    "MIKAN": lambda: MIKAN(
        config["widths"],
        edge_mlp_d=config["edge_mlp_hidden_d"],
        activation=ACTIVATION[config["edge_mlp_activation"]],
        use_layernorm=config["use_layernorm"]
    ).to(device),
    "SharedMIKAN_EdgeWiseEmbedding": lambda: SharedMIKANEdgeWiseEmb(
        config["widths"],
        shared_edge_mlp_hidden_widths=config["shared_edge_mlp_hidden_widths"],
        embedding_dim=config["embedding_dim"],
        embedding_init_std=config["embedding_init_std"],
        activation=ACTIVATION[config["edge_mlp_activation"]],
        use_layernorm=config["use_layernorm"]
    ).to(device),
    "SharedMIKAN_NodeWiseEmbedding": lambda: SharedMIKANNodeWiseEmb(
        config["widths"],
        shared_edge_mlp_hidden_widths=config["shared_edge_mlp_hidden_widths"],
        in_embedding_dim=config["in_embedding_dim"],
        out_embedding_dim=config["out_embedding_dim"],
        embedding_init_std=config["embedding_init_std"],
        activation=ACTIVATION[config["edge_mlp_activation"]],
        use_layernorm=config["use_layernorm"]
    ).to(device),
    "SharedMIKAN_NodeWiseEmbeddingWithMixing": lambda: SharedMIKANNodeWiseEmb(
        config["widths"],
        shared_edge_mlp_hidden_widths=config["shared_edge_mlp_hidden_widths"],
        in_embedding_dim=config["in_embedding_dim"],
        out_embedding_dim=config["out_embedding_dim"],
        emb_mixing_mlp_hidden_output_widths=config["emb_mixing_mlp_hidden_output_widths"],
        embedding_init_std=config["embedding_init_std"],
        activation=ACTIVATION[config["edge_mlp_activation"]],
        use_layernorm=config["use_layernorm"]
    ).to(device),
}

OPTIMIZER = {
    "SGD": lambda params: optim.SGD(params, lr=config["learning_rate"]),
    "AdamW": lambda params: optim.AdamW(params, lr=config["learning_rate"]),
}


def train_on_fashion_mnist():

    if use_wandb:
            wandb.init(project=project_name, name=f"{config['model']}_{datetime.datetime.now()}", config=config, group=config["model"]+"_")
    
    batch_size = config["batch_size"]
    num_epoch = config["num_epoch"]

    train_dataset = datasets.FashionMNIST(
        './data',
        train = True,
        download = True,
        transform = ToTensor()
    )
    test_dataset = datasets.FashionMNIST(
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
    # for epoch in range(0):
        start = time.perf_counter()
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device, True)
        epoch_time = time.perf_counter() - start
        timer += epoch_time

        test_loss, test_acc = test_model(model, test_loader, criterion, device, True)
    
        print(f"Epoch {epoch}:")
        print(f"Train Accuracy: {train_acc}")
        print(f"Test Accuracy: {test_acc}")
        print(f"Time: {epoch_time:.4f} seconds\n")

        if use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "epoch_time": epoch_time
            })

    # Learn Embedding vectors only
    if config["model"] in ["SharedMIKAN_EdgeWiseEmbedding", "SharedMIKAN_NodeWiseEmbedding", "SharedMIKAN_NodeWiseEmbeddingWithMixing"]:
        print("Learning Embedding vectors only...")
        for param in model.parameters():
            param.requires_grad = False
        for layer in model.layers:
            if config["model"] == "SharedMIKAN_EdgeWiseEmbedding":
                layer.embedding.weight.requires_grad = True
            elif config["model"] in ["SharedMIKAN_NodeWiseEmbedding", "SharedMIKAN_NodeWiseEmbeddingWithMixing"]:
                layer.in_embedding.weight.requires_grad = True
                layer.out_embedding.weight.requires_grad = True
            if config["model"] == "SharedMIKAN_NodeWiseEmbeddingWithMixing":
                for param in layer.emb_mixing_mlp.parameters():
                    param.requires_grad = True

        optimizer = OPTIMIZER[config["optimizer"]](model.parameters())
        for epoch in range(config["embedding_only_num_epoch"]):
            start = time.perf_counter()
            train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device, True)
            epoch_time = time.perf_counter() - start
            timer += epoch_time

            test_loss, test_acc = test_model(model, test_loader, criterion, device, True)
        
            print(f"Embedding Epoch {epoch}:")
            print(f"Train Accuracy: {train_acc}")
            print(f"Test Accuracy: {test_acc}")
            print(f"Time: {epoch_time:.4f} seconds\n")

            if use_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "epoch_time": epoch_time
                })

                if epoch == config["embedding_only_num_epoch"] - 1:
                    wandb.summary["final_train_loss"] = train_loss
                    wandb.summary["final_train_acc"] = train_acc
                    wandb.summary["final_test_loss"] = test_loss
                    wandb.summary["final_test_acc"] = test_acc

    print(f"Time: {timer}")
    if use_wandb:
        wandb.summary["total_time"] = timer

    if use_wandb:
        wandb.finish()


def run_trials():
    # models = ["MLP-S", "MLP-L", "FastKAN", "FasterKAN", "MIKAN", "SharedMIKAN_EdgeWiseEmbedding", "SharedMIKAN_NodeWiseEmbedding", "SharedMIKAN_NodeWiseEmbeddingWithMixing"]
    models = ["SharedMIKAN_EdgeWiseEmbedding", "SharedMIKAN_NodeWiseEmbedding"]
    TRAINS_PER_MODEL = 1

    for model_name in models:
        config["model"] = model_name
        for _ in range(TRAINS_PER_MODEL):
            train_on_fashion_mnist()


def sweep_embedding_dim():
    global project_name
    project_name = "mikan-fashion-mnist-embedding-dim"
    models = ["SharedMIKAN_EdgeWiseEmbedding", "SharedMIKAN_NodeWiseEmbedding", "SharedMIKAN_NodeWiseEmbeddingWithMixing"]
    TRAINS_PER_MODEL = 10

    for model_name in models:
        config["model"] = model_name
        for embedding_dim in [4, 8, 16]:
            config["embedding_dim"] = embedding_dim
            config["in_embedding_dim"] = embedding_dim // 2
            config["out_embedding_dim"] = embedding_dim // 2
            for _ in range(TRAINS_PER_MODEL):
                train_on_fashion_mnist()


def sweep_shared_edge_mlp_hidden_widths():
    global project_name
    project_name = "mikan-fashion-mnist-shared-edge-mlp-hidden-widths"
    models = ["SharedMIKAN_EdgeWiseEmbedding", "SharedMIKAN_NodeWiseEmbedding", "SharedMIKAN_NodeWiseEmbeddingWithMixing"]
    TRAINS_PER_MODEL = 10

    for model_name in models:
        config["model"] = model_name
        for shared_edge_mlp_hidden_widths in [[4], [8], [32], [64]]:
            config["shared_edge_mlp_hidden_widths"] = shared_edge_mlp_hidden_widths
            for _ in range(TRAINS_PER_MODEL):
                train_on_fashion_mnist()


if __name__ == "__main__":
    # train_on_fashion_mnist()
    run_trials()
    # sweep_embedding_dim()
    # sweep_shared_edge_mlp_hidden_widths()
    