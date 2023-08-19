import math
from utils.logger import create_logger

import torch
import torch.nn as nn


class AbsolutePositionalEncoding(nn.Module):
    """
    A basic representation of Absolute (Sinusoidal) Positional Encoding defined
    in the 'Attention Is All You Need' paper (https://arxiv.org/abs/1706.03762).

    :param embed_dim: (int) the number of dimensions for each embedding (e.g., 512)
    :param drop_prob: (float) the neuron dropout probability
    :param max_len: (int, optional) the maximum size of the encodings (sequence length). Default is `1500`
    :param log_info: (bool, optional) a flag for logging information. Defaults to `False`
    """
    def __init__(self, embed_dim: int, drop_prob: float, max_len: int = 1500, log_info: bool = False) -> None:
        super().__init__()
        logger = create_logger('ape', filename='encoding', flag=log_info)
        self.dropout = nn.Dropout(p=drop_prob)

        # Compute positional encodings
        pe = torch.zeros(max_len, embed_dim)

        # Create tensor representing positions from 0 to max_len
        position = torch.arange(0, end=max_len).unsqueeze(1)
        logger.info(f'Positions: {position.size()}\n  {position}')

        # Compute division term for sinusoidal calculations
        div_term = torch.exp(torch.arange(0, end=embed_dim, step=2) * -(math.log(10000.0) / embed_dim))
        logger.info(f'Division value: {div_term}')

        # Calculate sine and cosine positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Add encodings to a buffer to ignore optimizer training
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional information to a set of embeddings (x).

        :param x: (torch.Tensor) input embeddings
        :return: (torch.Tensor) embeddings with positional information
        """
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)
