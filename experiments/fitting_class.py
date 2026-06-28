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
        flatten=False
    ):

    model.train()
    total_loss, correct = 0.0, 0.0

    for inputs, labels in tqdm(train_loader, desc="Trainig", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        if flatten:
            inputs = inputs.view(inputs.size(0), -1)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(axis=1)
        correct += torch.sum(preds == labels).item() / len(labels)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / len(train_loader)

    return avg_loss, accuracy


def test_model(
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        flatten=False
    ):

    model.eval()
    total_loss, correct = 0.0, 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            if flatten:
                inputs = inputs.view(inputs.size(0), -1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(axis=1)
            correct += torch.sum(preds == labels).item() / len(labels)

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / len(test_loader)

        return avg_loss, accuracy
