import math
from utils.logger import create_logger

import torch
import torch.nn as nn


def self_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                   mask: torch.Tensor = None, dropout_layer: nn.Module = None, log_info: bool = False) -> torch.Tensor:
    """
    Performs 'Scaled Dot Product Attention' from the 'Attention Is All You Need' paper (https://arxiv.org/abs/1706.03762).

    :param queries: (torch.Tensor) the queries vector
    :param keys: (torch.Tensor) the keys vector
    :param values: (torch.Tensor) the values vector
    :param mask: (torch.Tensor, optional) a mask to apply to the scaled scores. Defaults to `None`
    :param dropout_layer: (nn.Module, optional) a dropout layer to applying to the attention weights. Defaults to `None`
    :param log_info: (bool, optional) a flag for logging information. Defaults to `False`
    """
    logger = create_logger('self_attention', filename='self_attention', flag=log_info)
    d_k = keys.size(-1)  # number of tokens/key dimension

    # Scores: For each token, determine the attention focus against all other tokens
    scores = torch.matmul(queries, keys.transpose(-2, -1))
    scaled_scores = scores / math.sqrt(d_k)

    # Mask scores if required
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)  # -1e9: not infinity but large enough!

    attn_weights = torch.softmax(scaled_scores, dim=-1)

    logger.info(f'Scaled scores: {scaled_scores.size()} \n{scaled_scores}\n')
    logger.info(f'Attention weights (without dropout): {attn_weights.size()} \n{attn_weights}\n')

    if dropout_layer is not None:
        attn_weights = dropout_layer(attn_weights)
        logger.info(f'Attention weights (with dropout): {attn_weights.size()} \n{attn_weights}\n')

    # Attention output
    return torch.matmul(attn_weights, values)


class MultiHeadedAttention(nn.Module):
    """A basic representation of a Multi-Headed Attention module for Transformers.

    :param embed_size: (int) number of expected features
    :param n_heads: (int) number of heads in the multi-headed attention module
    :param drop_prob: (float, optional) the neuron dropout probability. Default is `0.1`
    :param log_info: (bool, optional) flag for enabling logging. Default is `False`
    """
    def __init__(self, embed_size: int, n_heads: int, drop_prob: float = 0.1, log_info: bool = False) -> None:
        super().__init__()
        self.logger = create_logger(self.__class__.__name__, filename='multi_attention', flag=log_info)

        self.log_info = log_info
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.head_dim = embed_size // n_heads

        assert (self.head_dim * n_heads == embed_size), ValueError("'embed_size' must be divisible by 'n_heads'")

        # Same layers, different weights
        self.fc_query = nn.Linear(self.embed_size, self.embed_size)
        self.fc_key = nn.Linear(self.embed_size, self.embed_size)
        self.fc_value = nn.Linear(self.embed_size, self.embed_size)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

        self.attention = None  # Store attention
        self.dropout = nn.Dropout(p=drop_prob)

    def split_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Splits a set of data (query, key, or values) into separate attention heads, going from 3 to 4 dimensions."""
        self.logger.info(f'Before head split: {x.size()}')  # [64, 100, 512]
        batch_size, seq_len, _ = x.size()  # [batch_size, seq_len, embed_size]
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combines the attention output heads into one, reducing it from 4 dimensions into 3. Returns the updated output."""
        self.logger.info(f'Before combining heads: {x.size()}')  # [64, 8, 100, 64]
        batch_size, _, seq_len, head_dim = x.size()  # [batch_size, n_heads, seq_len, head_dim]
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size)  # [batch_size, seq_len, embed_size]

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Performs a forward-pass through the Multi-Headed Attention process."""
        # Create respective query, key, value pairs (same input for each one)
        query = self.split_to_heads(self.fc_query(query))
        key = self.split_to_heads(self.fc_key(key))
        value = self.split_to_heads(self.fc_value(value))
        self.logger.info(f'Query, key, value dims (should be identical): '
                     f'\n  Query: {query.size()}'
                     f'\n  Key: {key.size()}'
                     f'\n  Value: {value.size()}')  # [64, 8, 100, 64]

        # Compute self attention
        attn_out = self_attention(queries=query, keys=key, values=value,
                                  mask=mask, dropout_layer=self.dropout,
                                  log_info=self.log_info)  # [batch_size, n_heads, seq_len, head_dim]
        self.logger.info(f'Attention output: {attn_out.size()}')  # [64, 8, 100, 64]

        # Combine the attention heads
        self.attention = self.combine_heads(attn_out)
        self.logger.info(f'After combining heads: {self.attention.size()}\n')  # [64, 100, 512]

        # Pass the combined heads through a final FC layer
        return self.fc_out(self.attention)  # [batch_size, seq_len, embed_size]
