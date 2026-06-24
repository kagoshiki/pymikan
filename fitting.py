import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from tqdm import tqdm
import time


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device, 
        disable_tqdm: bool = False
    ):

    model.train()
    total_loss, correct = 0.0, 0.0

    for inputs, labels in tqdm(train_loader, desc="Trainig", leave=False, disable=disable_tqdm):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(axis=1)
        correct += torch.sum(preds == labels).item() / len(labels)

    avg_loss = total_loss / len(train_loader)

    return avg_loss


def test_model(
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        disable_tqdm: bool = False
    ):

    model.eval()
    total_loss, correct = 0.0, 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False, disable=disable_tqdm):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(axis=1)
            correct += torch.sum(preds == labels).item() / len(labels)

        avg_loss = total_loss / len(test_loader)

        return avg_loss
