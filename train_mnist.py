import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from mlp import *
from fastkan import *
from fasterkan import *
from mikan.mikan import MIKAN

from tqdm import tqdm
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def train_model(model, train_loader, optimizer, criterion, device, flatten=False):
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


def test_model(model, test_loader, criterion, device, flatten=False):
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


batch_size = 64
num_hidden = 64
learning_rate = 0.01
num_epoch = 20

train_dataset = datasets.MNIST(
    './data',
    train = True,
    download = True,
    transform = ToTensor()
    )
test_dataset = datasets.MNIST(
    './data',
    train = False,
    download=True,
    transform = ToTensor()
    )

# num_trainset = 4000
# num_testset = 2000
# train_dataset = torch.utils.data.Subset(train_dataset, list(range(num_trainset)))
# test_dataset = torch.utils.data.Subset(test_dataset, list(range(num_testset)))

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

# model = MLP([784, num_hidden, 10]).to(device)
# model = FastKAN([784, num_hidden, 10], num_grids=10).to(device)
model = FasterKAN([784, num_hidden, 10], num_grids=10).to(device)
# model = MIKAN([784, num_hidden, 10], 2, F.relu).to(device)

criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)

timer = 0
for epoch in range(num_epoch):
    start = time.perf_counter()
    train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device, True)
    timer += time.perf_counter() - start

    test_loss, test_acc = test_model(model, test_loader, criterion, device, True)
   
    print(f"Epoch {epoch}:")
    print(f"Train Accuracy: {train_acc}")
    print(f"Test Accuracy: {test_acc}\n")

print(f"Time: {timer}")
