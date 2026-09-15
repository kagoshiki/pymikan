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

project_name = "mikan-adult-embonly"

config = {
    # common
    "model": "SharedMIKAN_EdgeWiseEmbedding",
    "batch_size": 64,
    "widths": [104, 64, 2],
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "num_epoch": 10,
    "embedding_only_num_epoch": 0,
    "validation_fraction": 0.1,
    "random_seed": 42,
    "early_stopping_patience": 3,
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
    "MLP": lambda: MLP([104, 64, 2], hidden_activation=ACTIVATION[config["hidden_activation"]]).to(device),
    "MLP-S" : lambda: MLP([104, 24, 2], hidden_activation=ACTIVATION[config["hidden_activation"]]).to(device),
    "MLP-L" : lambda: MLP([104, 800, 2], hidden_activation=ACTIVATION[config["hidden_activation"]]).to(device),
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


ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]
ADULT_NUMERIC_COLUMNS = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
    "hours-per-week"
]
ADULT_CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]


def _download_if_missing(path, url):
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".download")
    urllib.request.urlretrieve(url, temporary_path)
    temporary_path.replace(path)


@lru_cache(maxsize=1)
def load_adult_datasets():
    data_dir = Path(__file__).resolve().parents[2] / "data" / "Adult"
    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"
    _download_if_missing(
        train_path,
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    )
    _download_if_missing(
        test_path,
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    )

    train_frame = pd.read_csv(
        train_path,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_values="?"
    ).dropna()
    test_frame = pd.read_csv(
        test_path,
        names=ADULT_COLUMNS,
        skiprows=1,
        skipinitialspace=True,
        na_values="?"
    ).dropna()

    train_labels = (train_frame.pop("income") == ">50K").astype("int64")
    test_labels = (test_frame.pop("income").str.rstrip(".") == ">50K").astype("int64")

    for column in ADULT_NUMERIC_COLUMNS:
        mean = train_frame[column].mean()
        std = train_frame[column].std(ddof=0)
        train_frame[column] = (train_frame[column] - mean) / std
        test_frame[column] = (test_frame[column] - mean) / std

    for column in ADULT_CATEGORICAL_COLUMNS:
        categories = sorted(train_frame[column].unique())
        category_dtype = pd.CategoricalDtype(categories=categories)
        train_frame[column] = train_frame[column].astype(category_dtype)
        test_frame[column] = test_frame[column].astype(category_dtype)

    train_frame = pd.get_dummies(
        train_frame,
        columns=ADULT_CATEGORICAL_COLUMNS,
        dtype="float32"
    )
    test_frame = pd.get_dummies(
        test_frame,
        columns=ADULT_CATEGORICAL_COLUMNS,
        dtype="float32"
    )

    if train_frame.shape[1] != config["widths"][0]:
        raise ValueError(
            f"Expected {config['widths'][0]} Adult features, got {train_frame.shape[1]}"
        )

    train_dataset = TensorDataset(
        torch.from_numpy(train_frame.to_numpy(dtype="float32")),
        torch.from_numpy(train_labels.to_numpy())
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_frame.to_numpy(dtype="float32")),
        torch.from_numpy(test_labels.to_numpy())
    )

    return train_dataset, test_dataset


def train_on_adult():
    if use_wandb:
        wandb.init(project=project_name, name=f"{config['model']}_{datetime.datetime.now()}", config=config, group=config["model"])

    batch_size = config["batch_size"]
    num_epoch = config["num_epoch"]

    train_dataset, test_dataset = load_adult_datasets()
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
            model, train_loader, optimizer, criterion, device, True, num_classes=2
        )
        epoch_time = time.perf_counter() - start
        timer += epoch_time

        val_loss, val_acc, val_macro_f1 = test_model(
            model, validation_loader, criterion, device, True, num_classes=2
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
                model, train_loader, optimizer, criterion, device, True, num_classes=2
            )
            epoch_time = time.perf_counter() - start
            timer += epoch_time

            val_loss, val_acc, val_macro_f1 = test_model(
                model, validation_loader, criterion, device, True, num_classes=2
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
        model, test_loader, criterion, device, True, num_classes=2
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
    # models = ["MLP-S", "MLP-L", "KAN", "FastKAN", "FasterKAN", "MIKAN", "SharedMIKAN_EdgeWiseEmbedding", "SharedMIKAN_NodeWiseEmbedding", "SharedMIKAN_NodeWiseEmbeddingWithMixing"]
    models = ["MLP-S", "MLP-L"]
    TRAINS_PER_MODEL = 5

    for model_name in models:
        config["model"] = model_name
        for _ in range(TRAINS_PER_MODEL):
            train_on_adult()


def sweep_embedding_dim():
    global project_name
    project_name = "mikan-adult-embedding-dim"
    models = ["SharedMIKAN_EdgeWiseEmbedding", "SharedMIKAN_NodeWiseEmbedding", "SharedMIKAN_NodeWiseEmbeddingWithMixing"]
    TRAINS_PER_MODEL = 10

    for model_name in models:
        config["model"] = model_name
        for embedding_dim in [4, 8, 16]:
            config["embedding_dim"] = embedding_dim
            config["in_embedding_dim"] = embedding_dim // 2
            config["out_embedding_dim"] = embedding_dim // 2
            for _ in range(TRAINS_PER_MODEL):
                train_on_adult()


def sweep_shared_edge_mlp_hidden_widths():
    global project_name
    project_name = "mikan-adult-shared-edge-mlp-hidden-widths"
    models = ["SharedMIKAN_EdgeWiseEmbedding", "SharedMIKAN_NodeWiseEmbedding", "SharedMIKAN_NodeWiseEmbeddingWithMixing"]
    TRAINS_PER_MODEL = 10

    for model_name in models:
        config["model"] = model_name
        for shared_edge_mlp_hidden_widths in [[4], [8], [32], [64]]:
            config["shared_edge_mlp_hidden_widths"] = shared_edge_mlp_hidden_widths
            for _ in range(TRAINS_PER_MODEL):
                train_on_adult()


if __name__ == "__main__":
    # train_on_adult()
    run_trials()
    # sweep_embedding_dim()
    # sweep_shared_edge_mlp_hidden_widths()
