import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    """
    A module dedicated to Layer Normalization formulated in the paper https://arxiv.org/abs/1607.06450.

    :param feature_dims: (int) the number of input features or channels
    :param epsilon: (float, optional) a value added to the denominator for numerical stability. Default is `1e-6`
    :param device: (string, optional) name of the PyTorch CUDA device to connect to (if CUDA is available). Defaults to `cpu`
    """
    def __init__(self, feature_dims: int, epsilon: float = 1e-6, device: str = 'cpu') -> None:
        super().__init__()
        self.device = device

        self.feature_dims = feature_dims
        self.gain = nn.Parameter(torch.ones(feature_dims)).to(self.device)  # gamma
        self.bias = nn.Parameter(torch.zeros(feature_dims)).to(self.device)  # beta
        self.eps = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Layer Normalization to a torch.Tensor of data."""
        x = x.to(self.device)

        # Calculate mean and variance along feature dimension
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(variance + self.eps)

        return self.gain * x_norm + self.bias  # scale and shift


class ResidualAndLayerNorm(nn.Module):
    """
    Performs an intermediate step between each module in the Transformer.

    :param feature_dims: (int) the number of input features or channels
    :param drop_prob: (float, optional) the neuron dropout probability. Defaults to `0.1`
    :param epsilon: (float, optional) a value added to the normalization for numerical stability. Default: `cpu`
    :param device: (string, optional) name of the PyTorch CUDA device to connect to (if CUDA is available). Defaults to `cpu`
    """
    def __init__(self, feature_dims: int, drop_prob: float = 0.1,
                 epsilon: float = 1e-6, device: str = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.norm = LayerNormalization(feature_dims=feature_dims, epsilon=epsilon).to(self.device)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor, module: nn.Module) -> torch.Tensor:
        """
        Computes an output from a given module using x. X is normalised first (instead of last, for simplicity)
        and then passed through the module and added to the residual connection (x).

        :param x: (torch.Tensor) input data
        :param module: (nn.Module) a torch module associated to a Transformer module
                       (E.g., Multi-Headed Attention or a Feed-Forward Network)
        """
        x = x.to(self.device)
        return x + self.dropout(module(self.norm(x)))
