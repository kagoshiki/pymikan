import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import wandb
import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

use_wandb = False

config = {
    "num_func": 12,
    "embedding_dim": 2,
    "hidden_dim": 64,
    "num_samples": 1000,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "activation": "tanh"
}

if use_wandb:
    wandb.init(
        project="multi-function-approximation",
        name=f"mfa-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        config=config
    )


# Multi-Function Approximation using Multi-Layer Perceptron (MLP)
# [1 + embedding_dim, hidden_dim, 1]
class MFAMLP(nn.Module):
    def __init__(self, num_func: int, embedding_dim: int, hidden_dim: int, activation: callable = F.relu):
        super(MFAMLP, self).__init__()
        self.num_func = num_func
        self.embedding = nn.Embedding(num_func, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.activation = activation
    
    def forward(self, x: torch.Tensor, func_id: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, 1]
        # func_id: [batch_size]
        if x.size(0) != func_id.size(0):
            raise ValueError("Batch size of x and func_id must match.")
        
        embed = self.embedding(func_id)  # [batch_size, embedding_dim]
        x = torch.cat([x, embed], dim=1)  # [batch_size, embedding_dim + 1]
        x = self.activation(self.fc1(x))  # [batch_size, hidden_dim]
        output = self.fc2(x)  # [batch_size, 1]
        return output


# Dataset for Multi-Function Approximation
class MathFunctionDataset(Dataset):
    def __init__(self, start: float, end: float, num_samples: int, func: callable):
        super().__init__()
        self.x = torch.linspace(start, end, num_samples).unsqueeze(1)  # [num_samples, 1]
        self.y = func(self.x)  # [num_samples, 1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Experiment to train the MLP on multiple functions
def train_mfa_mlp():
    # Define multiple functions to approximate
    functions = [
        lambda x: x,
        lambda x: -x,
        lambda x: torch.abs(x),
        lambda x: torch.sin(x * torch.pi),
        lambda x: -torch.sin(x * torch.pi),
        lambda x: torch.cos(x * torch.pi),
        lambda x: torch.tanh(x * torch.pi),
        lambda x: torch.exp(x - 1),
        lambda x: x ** 2,
        lambda x: x ** 3,
        lambda x: x ** 4,
        lambda x: 0 * x,
    ]

    function_names = [
        "y=x",
        "y=-x",
        "y=|x|",
        "y=sin(πx)",
        "y=-sin(πx)",
        "y=cos(πx)",
        "y=tanh(πx)",
        "y=exp(x-1)",
        "y=x^2",
        "y=x^3",
        "y=x^4",
        "y=0"
    ]

    num_func = len(functions)
    embedding_dim = config["embedding_dim"]
    hidden_dim = config["hidden_dim"]
    num_samples = config["num_samples"]
    num_epochs = config["num_epochs"]

    # Create datasets and dataloaders for each function
    datasets = [MathFunctionDataset(-1, 1, num_samples, func) for func in functions]
    dataloaders = [DataLoader(dataset, batch_size=1, shuffle=True) for dataset in datasets]
    
    # Initialize the MLP model
    activation = None
    if config["activation"] == "relu":
        activation = F.relu
    elif config["activation"] == "tanh":
        activation = F.tanh
    elif config["activation"] == "sigmoid":
        activation = F.sigmoid
    else:
        activation = F.relu
    model = MFAMLP(num_func=num_func, embedding_dim=embedding_dim, hidden_dim=hidden_dim, activation=activation)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training MFA MLP"):
        total_loss = 0.0
        
        iters = zip(*[iter(dl) for dl in dataloaders])
    
        for batches in iters:
            x = torch.stack([b[0] for b in batches]).squeeze(1).to(device)     # [4, 1]
            y = torch.stack([b[1] for b in batches]).squeeze(1).to(device)     # [4, 1]
            func_ids = torch.arange(num_func).to(device)                       # [4]

            optimizer.zero_grad()
            outputs = model(x, func_ids)                      # [4, 1]
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloaders[0])
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.8f}")
        if use_wandb:
            wandb.log({"loss": avg_loss})

    # Evaluate loss and save plots for each function
    print("\nEvaluating and plotting results...")
    fig, ax = plt.subplots(3, 4, figsize=(20, 15))
    for i, func in enumerate(functions):
        x = torch.linspace(-1, 1, 100).unsqueeze(1).to(device)
        y_true = func(x)
        y_pred = model(x, torch.tensor([i] * len(x)).to(device))

        mse_loss = criterion(y_pred, y_true).item()
        print(f"Function: {function_names[i]}, MSE Loss: {mse_loss:.8f}")
        
        plt.figure()
        plt.plot(x.cpu().numpy(), y_true.cpu().numpy(), label="Ground Truth")
        plt.plot(x.cpu().numpy(), y_pred.detach().cpu().numpy(), label="MLP Approximation")
        plt.title(f"{function_names[i]}")
        if i == len(functions) - 1:
            plt.ylim(-1.0, 1.0)
        plt.legend()
        # plt.savefig(f"results/function_{i + 1}.svg", format="svg", bbox_inches="tight")
        plt.close()

        ax[i // 4, i % 4].plot(x.cpu().numpy(), y_true.cpu().numpy(), label="Ground Truth")
        ax[i // 4, i % 4].plot(x.cpu().numpy(), y_pred.detach().cpu().numpy(), label="MLP Approximation")
        ax[i // 4, i % 4].set_title(f"{function_names[i]}\nMSE: {mse_loss:.8f}")
        if i == len(functions) - 1:
            ax[i // 4, i % 4].set_ylim(-1.0, 1.0)
        if i == 0:
            ax[i // 4, i % 4].legend()

        if use_wandb:
            wandb.log({
                f"plot_{i}": wandb.plot.line_series(
                    x.cpu().numpy().flatten(), 
                    [y_true.cpu().numpy().flatten(), y_pred.detach().cpu().numpy().flatten()],
                    keys=["Ground Truth", "MLP Approximation"],
                    title=function_names[i]
                )
            })
    fig.savefig("results/all_functions.eps", bbox_inches="tight")

    # Output Embedding Vectors
    embeddings = model.embedding.weight.data.cpu().numpy()
    np.savetxt("results/embeddings.csv", embeddings, delimiter=",")
    if embeddings.shape[1] == 2:
        plt.figure()
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=np.arange(num_func), cmap="tab10")
        for i in range(num_func):
            plt.text(embeddings[i, 0] + 0.01, embeddings[i, 1], function_names[i], fontsize=9)
        plt.title("Embedding Vectors")
        plt.grid(True)
        plt.savefig("results/function_embeddings.svg", format="svg", bbox_inches="tight")
        plt.close()

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train_mfa_mlp()
