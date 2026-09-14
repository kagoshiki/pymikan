import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from functools import lru_cache
import urllib.request

import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

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

from src.experiments.fitting_class_macro_f1 import EarlyStopping, train_model, test_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

use_wandb = True

project_name = "mikan-covertype-embonly"

config = {
    # common
    "model": "KAN",
    "batch_size": 64,
    "widths": [54, 64, 7],
    "optimizer": "AdamW",
    "learning_rate": 0.005,
    "num_epoch": 50,
    "embedding_only_num_epoch": 0,
    "validation_fraction": 0.1,
    "random_seed": 42,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.001,
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
    "MLP": lambda: MLP([54, 64, 7], hidden_activation=ACTIVATION[config["hidden_activation"]]).to(device),
    "MLP-S" : lambda: MLP([54, 8, 7], hidden_activation=ACTIVATION[config["hidden_activation"]]).to(device),
    "MLP-L" : lambda: MLP([54, 54, 7], hidden_activation=ACTIVATION[config["hidden_activation"]]).to(device),
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


@lru_cache(maxsize=1)
def load_covertype_datasets():
    data_path = Path(__file__).resolve().parents[2] / "data" / "covertype" / "covtype.data.gz"
    _download_if_missing(
        data_path,
        "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    )

    frame = pd.read_csv(data_path, header=None, compression="gzip", dtype="float32")
    features = torch.from_numpy(frame.iloc[:, :-1].to_numpy(dtype="float32"))
    labels = torch.from_numpy(frame.iloc[:, -1].to_numpy(dtype="int64")) - 1

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(features), generator=generator)
    train_size = int(0.8 * len(features))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_features = features[train_indices]
    test_features = features[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    mean = train_features[:, :10].mean(dim=0)
    std = train_features[:, :10].std(dim=0, correction=0)
    train_features[:, :10] = (train_features[:, :10] - mean) / std
    test_features[:, :10] = (test_features[:, :10] - mean) / std

    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    return train_dataset, test_dataset


def _download_if_missing(path, url):
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".download")
    urllib.request.urlretrieve(url, temporary_path)
    temporary_path.replace(path)


def train_on_covertype():
    if use_wandb:
        wandb.init(project=project_name, name=f"{config['model']}_{datetime.datetime.now()}", config=config, group=config["model"])

    batch_size = config["batch_size"]
    num_epoch = config["num_epoch"]

    train_dataset, test_dataset = load_covertype_datasets()
    validation_size = int(len(train_dataset) * config["validation_fraction"])
    train_size = len(train_dataset) - validation_size
    train_dataset, validation_dataset = random_split(
        train_dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(config["random_seed"])
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
    validation_loader = DataLoader(
        validation_dataset,
        batch_size = batch_size,
        shuffle = False
    )

    model = MODEL[config["model"]]()

    criterion = nn.CrossEntropyLoss()

    optimizer = OPTIMIZER[config["optimizer"]](model.parameters())
    early_stopping = EarlyStopping(
        patience=config["early_stopping_patience"],
        min_delta=config["early_stopping_min_delta"]
    )

    timer = 0
    best_epoch = 0
    best_metrics = None
    for epoch in range(num_epoch):
        start = time.perf_counter()
        train_loss, train_acc, train_macro_f1 = train_model(
            model, train_loader, optimizer, criterion, device, True, num_classes=7
        )
        epoch_time = time.perf_counter() - start
        timer += epoch_time

        val_loss, val_acc, val_macro_f1 = test_model(
            model, validation_loader, criterion, device, True, num_classes=7
        )

        print(f"Epoch {epoch}:")
        print(f"Train Loss: {train_loss}")
        print(f"Train Accuracy: {train_acc}")
        print(f"Train Macro F1: {train_macro_f1}")
        print(f"Validation Loss: {val_loss}")
        print(f"Validation Accuracy: {val_acc}")
        print(f"Validation Macro F1: {val_macro_f1}")
        print(f"Time: {epoch_time:.4f} seconds\n")

        if use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_macro_f1": train_macro_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "epoch_time": epoch_time
            })

        previous_best_loss = early_stopping.best_loss
        should_stop = early_stopping.update(val_loss, model)
        if early_stopping.best_loss < previous_best_loss:
            best_epoch = epoch
            best_metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_macro_f1": train_macro_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1
            }
        if should_stop:
            print(f"Early stopping at epoch {epoch}; best epoch: {best_epoch}")
            break

    early_stopping.restore_best_weights(model)

    # Learn Embedding vectors only
    if config["embedding_only_num_epoch"] > 0 and config["model"] in ["SharedMIKAN_EdgeWiseEmbedding", "SharedMIKAN_NodeWiseEmbedding", "SharedMIKAN_NodeWiseEmbeddingWithMixing"]:
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
            train_loss, train_acc, train_macro_f1 = train_model(
                model, train_loader, optimizer, criterion, device, True, num_classes=7
            )
            epoch_time = time.perf_counter() - start
            timer += epoch_time

            val_loss, val_acc, val_macro_f1 = test_model(
                model, validation_loader, criterion, device, True, num_classes=7
            )

            print(f"Embedding Epoch {epoch}:")
            print(f"Train Loss: {train_loss}")
            print(f"Train Accuracy: {train_acc}")
            print(f"Train Macro F1: {train_macro_f1}")
            print(f"Validation Loss: {val_loss}")
            print(f"Validation Accuracy: {val_acc}")
            print(f"Validation Macro F1: {val_macro_f1}")
            print(f"Time: {epoch_time:.4f} seconds\n")

            if use_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_macro_f1": train_macro_f1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_macro_f1": val_macro_f1,
                    "epoch_time": epoch_time
                })

    test_loss, test_acc, test_macro_f1 = test_model(
        model, test_loader, criterion, device, True, num_classes=7
    )
    print("Test:")
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")
    print(f"Test Macro F1: {test_macro_f1}\n")

    print(f"Time: {timer}")
    if use_wandb:
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_macro_f1": test_macro_f1
        })
        wandb.summary["best_epoch"] = best_epoch
        if best_metrics is not None:
            for name, value in best_metrics.items():
                wandb.summary[f"best_{name}"] = value
        wandb.summary["final_test_loss"] = test_loss
        wandb.summary["final_test_acc"] = test_acc
        wandb.summary["final_test_macro_f1"] = test_macro_f1
        wandb.summary["total_time"] = timer

    if use_wandb:
        wandb.finish()


def run_trials():
    models = ["MLP-S", "MLP-L", "KAN", "FastKAN", "FasterKAN", "MIKAN", "SharedMIKAN_EdgeWiseEmbedding", "SharedMIKAN_NodeWiseEmbedding", "SharedMIKAN_NodeWiseEmbeddingWithMixing"]
    # models = ["KAN"]
    TRAINS_PER_MODEL = 5

    for model_name in models:
        config["model"] = model_name
        for _ in range(TRAINS_PER_MODEL):
            train_on_covertype()


def sweep_embedding_dim():
    global project_name
    project_name = "mikan-covertype-embedding-dim"
    models = ["SharedMIKAN_EdgeWiseEmbedding", "SharedMIKAN_NodeWiseEmbedding", "SharedMIKAN_NodeWiseEmbeddingWithMixing"]
    TRAINS_PER_MODEL = 10

    for model_name in models:
        config["model"] = model_name
        for embedding_dim in [4, 8, 16]:
            config["embedding_dim"] = embedding_dim
            config["in_embedding_dim"] = embedding_dim // 2
            config["out_embedding_dim"] = embedding_dim // 2
            for _ in range(TRAINS_PER_MODEL):
                train_on_covertype()


def sweep_shared_edge_mlp_hidden_widths():
    global project_name
    project_name = "mikan-covertype-shared-edge-mlp-hidden-widths"
    models = ["SharedMIKAN_EdgeWiseEmbedding", "SharedMIKAN_NodeWiseEmbedding", "SharedMIKAN_NodeWiseEmbeddingWithMixing"]
    TRAINS_PER_MODEL = 10

    for model_name in models:
        config["model"] = model_name
        for shared_edge_mlp_hidden_widths in [[4], [8], [32], [64]]:
            config["shared_edge_mlp_hidden_widths"] = shared_edge_mlp_hidden_widths
            for _ in range(TRAINS_PER_MODEL):
                train_on_covertype()


if __name__ == "__main__":
    # train_on_covertype()
    run_trials()
    # sweep_embedding_dim()
    # sweep_shared_edge_mlp_hidden_widths()
