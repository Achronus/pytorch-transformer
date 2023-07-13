"""
A code example demo of Multi-Headed Attention. Uses the same code from the `/model` directory with the addition of print statements for tensor debugging.
Additionally, contains an example of PositionalEncoding and torch.nn.Embedding to demonstrate a full implementation.
"""

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
        print(f'Before split: {x.size()}')  # [64, 100, 512]
        batch_size, seq_len, _ = x.size()  # [batch_size, seq_len, embed_size]
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        print(f'Before combining: {x.size()}')  # [64, 8, 100, 64]
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
        print(f'Query, key, value dims (should be identical): '
              f'\n  {query.size()}'
              f'\n  {key.size()}'
              f'\n  {value.size()}')  # [64, 8, 100, 64]

        attn_out = self_attention(queries=query, keys=key, values=value, mask=mask)  # [batch_size, n_heads, seq_len, head_dim]
        print(f'Attention output: {attn_out.size()}')  # [64, 8, 100, 64]
        self.attention = self.combine_heads(attn_out)
        print(f'After combining: {self.attention.size()}')  # [64, 100, 512]
        return self.fc_out(self.attention)  # [batch_size, seq_len, embed_size]


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, max_seq_length: int) -> None:
        """A positional encoding example from `Arjun Sarkar's Medium Blog` post
        (https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb)."""
        super().__init__()

        pe = torch.zeros(max_seq_length, embed_size)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


if __name__ == "__main__":
    SRC_VOCAB_SIZE = 5000
    TGT_VOCAB_SIZE = 5000
    EMBED_SIZE = 512
    NUM_HEADS = 8
    MAX_SEQ_LENGTH = 100
    BATCH_SIZE = 64

    # Generate random sample data
    src_data = torch.randint(1, SRC_VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LENGTH))
    tgt_data = torch.randint(1, TGT_VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LENGTH))

    mh_attn = MultiHeadedAttention(embed_size=EMBED_SIZE, n_heads=NUM_HEADS)
    pos_encode = PositionalEncoding(embed_size=EMBED_SIZE, max_seq_length=MAX_SEQ_LENGTH)  # To be refactored/replaced
    embedding = nn.Embedding(SRC_VOCAB_SIZE, EMBED_SIZE)

    encoded_input = pos_encode(embedding(src_data))
    out = mh_attn.forward(encoded_input, encoded_input, encoded_input)  # [64, 100, 512] -> [batch_size, seq_len, embed_size]
    print(f'Final output: {out.size()}')

