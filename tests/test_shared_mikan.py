from mikan_shared import SharedMIKAN, SharedMIKANLayer
import torch
import torch.nn.functional as F
import pytest


def test_shared_mikan_layer_forward():
    input_dim = 4
    output_dim = 3
    edge_mlp_hidden_width = [8]
    embedding_dim = 4
    activation = F.relu

    layer = SharedMIKANLayer(input_dim, output_dim, edge_mlp_hidden_width, embedding_dim, activation)

    batch_size = 2
    x = torch.randn(batch_size, input_dim)

    output = layer(x)

    assert output.shape == (batch_size, output_dim), f"{output.shape} != {(batch_size, output_dim)}"
