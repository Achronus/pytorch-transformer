import torch
import torch.nn as nn


class PositionWiseFFN(nn.Module):
    """
    A simple Position-Wise Feed-Forward Network found in the Transformer architecture.

    :param embed_dim: (int) the input embedding size
    :param hidden_dim: (int) the number of nodes in the hidden layer
    :param drop_prob: (float, optional) the neuron dropout probability. Default is `0.1`
    """
    def __init__(self, embed_dim: int, hidden_dim: int, drop_prob: float = 0.1) -> None:
        super().__init__()
        # Set network structure
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes a set of embeddings through the network."""
        # x -> fc1 -> relu -> dropout -> fc2 -> output
        fc1_out = self.relu(self.fc1(x))
        return self.fc2(self.dropout(fc1_out))
