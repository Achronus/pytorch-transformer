import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    """
    A module dedicated to Layer Normalization formulated in the paper https://arxiv.org/abs/1607.06450.

    :param feature_dims: (tuple[int, ...] | torch.size) the shape of the data to normalize
    :param epsilon: (float, optional) a value added to the denominator for numerical stability. Default: 1e-6
    """
    def __init__(self, feature_dims: tuple[int, ...], epsilon: float = 1e-6) -> None:
        super().__init__()
        self.feature_dims = feature_dims
        self.gain = nn.Parameter(torch.ones(feature_dims))  # gamma
        self.bias = nn.Parameter(torch.zeros(feature_dims))  # beta
        self.eps = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Layer Normalization to a torch.Tensor of data."""
        dims = [-(i + 1) for i in range(len(self.feature_dims))]  # reverse dimensions

        mean = x.mean(dim=dims, keepdim=True)
        variance = x.var(dim=dims, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(variance + self.eps)

        return self.gain * x_norm + self.bias  # scale and shift


class ResidualAndLayerNorm(nn.Module):
    """
    Performs an intermediate step between each module in the Transformer.

    :param feature_dims: (tuple[int, ...] | torch.size) the shape of the data to normalize
    :param epsilon: (float, optional) a value added to the normalization for numerical stability. Default: 1e-6
    """
    def __init__(self, feature_dims: tuple[int, ...], epsilon: float = 1e-6) -> None:
        super().__init__()
        self.norm = LayerNormalization(feature_dims=feature_dims, epsilon=epsilon)

    def forward(self, x: torch.Tensor, module: nn.Module) -> torch.Tensor:
        """Computes an output from a torch.nn.Module using x, applies a residual connection and normalizes it."""
        return self.norm(x + module(x))
