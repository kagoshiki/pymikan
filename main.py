import torch
from torch import nn


def main():
    print("Hello from pymikan!")

def test_embedding():
    embedding = nn.Embedding(5, 3)
    print(embedding.weight)
    print(embedding(torch.tensor([1, 2, 3])))
    print(embedding(torch.tensor([i for i in range(5)])))

def test_sum():
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    print(a.sum(dim=0))
    print(a.sum(dim=1))

def test_repeat():
    a = torch.tensor([[1, 2], [3, 4]])
    print(a)
    print(a.repeat(2, 1))
    print(a.repeat(1, 2))
    print(a.repeat_interleave(2, 1))

def test_cat():
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    c = torch.zeros((3, 2, 2), dtype=torch.int)
    print(a)
    print(b)
    print(torch.cat([a, b], dim=0))
    print(torch.cat([a, b], dim=1))
    print(c)
    print(torch.cat([c, a.expand(3, -1, -1)], dim=-1))


if __name__ == "__main__":
    # main()
    # test_embedding()
    # test_sum()
    # test_repeat()
    test_cat()