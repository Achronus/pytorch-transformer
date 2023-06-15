import math

import torch
import torch.nn as nn


def self_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Performs 'Scaled Dot Product Attention' from the 'Attention Is All You Need' paper (https://arxiv.org/abs/1706.03762)."""
    d_k = keys.size(-1)  # number of tokens/key dimension

    # Scores: For each token, determine the attention focus against all other tokens
    scores = torch.matmul(queries, keys.transpose(-2, -1))
    scaled_scores = scores / math.sqrt(d_k)

    # Mask scores if required
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)  # -1e9: not infinity but large enough!

    attn_weights = torch.softmax(scaled_scores, dim=-1)

    # Attention output
    return torch.matmul(attn_weights, values)


class MultiHeadedAttention(nn.Module):
    """A basic representation of a Multi-Headed Attention module for Transformers.

    :param embed_size: (int) number of expected features
    :param n_heads: (int) number of heads in the multi-headed attention module
    """
    def __init__(self, embed_size: int, n_heads: int) -> None:
        super().__init__()

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

    def split_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()  # [batch_size, seq_len, embed_size]
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, head_dim = x.size()  # [batch_size, n_heads, seq_len, head_dim]
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size)  # [batch_size, seq_len, embed_size]

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            # Apply mask to all heads
            mask = mask.unsqueeze(1)  # 3-dimensional

        # Create respective query, key, value pairs (same input for each one)
        query = self.split_to_heads(self.fc_query(query))
        key = self.split_to_heads(self.fc_key(key))
        value = self.split_to_heads(self.fc_value(value))

        attn_out = self_attention(queries=query, keys=key, values=value, mask=mask)  # [batch_size, n_heads, seq_len, head_dim]
        self.attention = self.combine_heads(attn_out)
        return self.fc_out(self.attention)  # [batch_size, seq_len, embed_size]
