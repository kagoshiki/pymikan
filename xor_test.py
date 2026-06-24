import torch
import torch.nn as nn

from tqdm import tqdm

import random

from mlp import MLP
from fastkan import FastKAN
from fasterkan import FasterKAN
from mikan import MIKAN

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

success, fail = 0, 0
start_seed = random.randint(0, 10000)

for trial in range(100):
    torch.manual_seed(start_seed + trial)  # 毎回違う初期値
    # model = nn.Sequential(nn.Linear(2,4), nn.Tanh(), nn.Linear(4,1))
    # model = MLP([2, 4, 1], hidden_activation=torch.sigmoid)
    # model = FastKAN([2, 4, 1], num_grids=10)
    # model = FasterKAN([2, 4, 1], num_grids=10)
    model = MIKAN([2, 2, 1], edge_mlp_d=4, activation=torch.sigmoid)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in tqdm(range(1000), desc=f"Trial {trial+1}/100", leave=False):
        loss = loss_fn(model(X), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    with torch.no_grad():
        outputs = torch.sigmoid(model(X))
        pred = (outputs > 0.5).float()

    if pred.eq(y).all():
        success += 1
        print(outputs)
    else:
        fail += 1

print(f"成功: {success}/100, 失敗: {fail}/100")
