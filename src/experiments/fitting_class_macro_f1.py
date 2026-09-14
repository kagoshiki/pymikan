import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from tqdm import tqdm


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        flatten=False,
        num_classes=None
    ):

    model.train()
    total_loss, correct, total_samples = 0.0, 0, 0
    if num_classes is not None:
        confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for inputs, labels in tqdm(train_loader, desc="Trainig", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        if flatten:
            inputs = inputs.view(inputs.size(0), -1)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(axis=1)
        correct += torch.sum(preds == labels).item()
        total_samples += batch_size
        if num_classes is not None:
            confusion_matrix += torch.bincount(
                (labels * num_classes + preds).detach().cpu(),
                minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    if num_classes is None:
        return avg_loss, accuracy

    return avg_loss, accuracy, macro_f1_from_confusion_matrix(confusion_matrix)


def test_model(
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        flatten=False,
        num_classes=None
    ):

    model.eval()
    total_loss, correct, total_samples = 0.0, 0, 0
    if num_classes is not None:
        confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            if flatten:
                inputs = inputs.view(inputs.size(0), -1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(axis=1)
            correct += torch.sum(preds == labels).item()
            total_samples += batch_size
            if num_classes is not None:
                confusion_matrix += torch.bincount(
                    (labels * num_classes + preds).detach().cpu(),
                    minlength=num_classes ** 2
                ).reshape(num_classes, num_classes)

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples

        if num_classes is None:
            return avg_loss, accuracy

        return avg_loss, accuracy, macro_f1_from_confusion_matrix(confusion_matrix)


def macro_f1_from_confusion_matrix(confusion_matrix: torch.Tensor):
    confusion_matrix = confusion_matrix.to(torch.float64)
    true_positives = confusion_matrix.diag()
    false_positives = confusion_matrix.sum(dim=0) - true_positives
    false_negatives = confusion_matrix.sum(dim=1) - true_positives
    denominator = 2 * true_positives + false_positives + false_negatives
    f1_per_class = torch.where(
        denominator > 0,
        2 * true_positives / denominator,
        torch.zeros_like(denominator)
    )
    return f1_per_class.mean().item()


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_state_dict = None
        self.epochs_without_improvement = 0

    def update(self, validation_loss, model):
        if self.best_state_dict is None or validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.best_state_dict = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience

    def restore_best_weights(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
