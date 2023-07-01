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


if __name__ == "__main__":
    torch.manual_seed(6)
    sample = torch.rand([2, 3, 2, 4])
    print(f'Before: {sample.size()} \n{sample}')
    ln = LayerNormalization(sample.size())

    x = ln(sample)
    print(f'After ln: {x.size()} \n{x}')