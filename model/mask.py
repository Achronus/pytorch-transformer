import torch
import torch.nn as nn


def generate_mask(x: torch.Tensor, mask_value: int = -1e9) -> torch.Tensor:
    """Creates a mask containing 0s and a custom value on the upper diagonal. Mask is the same size as the given input."""
    # Create a temporary tensor that matches the input size
    temp = torch.ones(x.size())
    temp = torch.fill(temp, mask_value)  # Fill with custom value
    return torch.triu(temp, diagonal=1)  # Convert to mask and return
