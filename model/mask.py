import torch
import torch.nn as nn


def generate_mask(x: torch.Tensor) -> torch.Tensor:
    """Creates a mask containing 0s and 1s, where 0s are on the upper diagonal. Mask is the same size as the given input."""
    # Create a temporary tensor that matches the input size
    temp = torch.ones(x.size())
    return 1 - torch.triu(temp, diagonal=1)  # Convert to mask and return
