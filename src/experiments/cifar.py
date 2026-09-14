import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from src.models.mlp import MLP
from src.models.efficientkan import KAN
from src.models.fastkan import FastKAN
from src.models.fasterkan import FasterKAN
from src.models.mikan import MIKAN
from src.models.shared_mikan import SharedMIKANEdgeWiseEmb, SharedMIKANNodeWiseEmb
from src.models.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202

import wandb
from tqdm import tqdm
import time
import datetime

from src.experiments.fitting_class import train_model, test_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

use_wandb = True

project_name = "mikan-cifar"

config = {
    # common
    "model": "KAN",
    "batch_size": 128,
    "widths": [64, 32, 10],
    "optimizer": "SGD",
    "learning_rate": 0.1,
    "num_epoch": 150,
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
    "MLP": lambda: MLP([64, 32, 10], hidden_activation=ACTIVATION[config["hidden_activation"]]).to(device),
    "MLP-S" : lambda: MLP([64, 32, 10], hidden_activation=ACTIVATION[config["hidden_activation"]]).to(device),
    "MLP-L" : lambda: MLP([64, 32, 10], hidden_activation=ACTIVATION[config["hidden_activation"]]).to(device),
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
    "SGD": lambda params: optim.SGD(params, lr=config["learning_rate"], momentum=0.9, weight_decay=1e-4),
    "AdamW": lambda params: optim.AdamW(params, lr=config["learning_rate"]),
}


def train_on_cifar():
    if use_wandb:
        wandb.init(project=project_name, name=f"{config['model']}_{datetime.datetime.now()}", config=config, group=config["model"])

    batch_size = config["batch_size"]
    num_epoch = config["num_epoch"]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    test_dataset = datasets.CIFAR10(
        root='./data',
        train = False,
        download=True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
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

    classifier = MODEL[config["model"]]()
    model = resnet32().to(device)
    model.linear = classifier

    criterion = nn.CrossEntropyLoss()

    optimizer = OPTIMIZER[config["optimizer"]](model.parameters())

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100])

    timer = 0
    for epoch in range(num_epoch):
        start = time.perf_counter()
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device, False)
        epoch_time = time.perf_counter() - start
        lr_scheduler.step()
        timer += epoch_time

        test_loss, test_acc = test_model(model, test_loader, criterion, device, False)
    
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

            if epoch == num_epoch - 1:
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
    models = ["MLP-S", "MLP-L", "KAN", "FastKAN", "FasterKAN", "MIKAN", "SharedMIKAN_EdgeWiseEmbedding", "SharedMIKAN_NodeWiseEmbedding", "SharedMIKAN_NodeWiseEmbeddingWithMixing"]
    # models = ["MLP-S", "MLP-L"]
    TRAINS_PER_MODEL = 5

    for model_name in models:
        config["model"] = model_name
        for _ in range(TRAINS_PER_MODEL):
            train_on_cifar()


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
                train_on_cifar()


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
                train_on_cifar()


if __name__ == "__main__":
    # train_on_cifar()
    run_trials()
    # sweep_embedding_dim()
    # sweep_shared_edge_mlp_hidden_widths()
    