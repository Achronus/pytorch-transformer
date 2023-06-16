import math

import torch
import torch.nn as nn


def generate_mask(x: torch.Tensor) -> torch.Tensor:
    """Creates a mask containing 0s and 1s, where 0s are on the upper diagonal. Mask is the same size as the given input."""
    # Create a temporary tensor that matches the input size
    temp = torch.ones(x.size())
    return 1 - torch.triu(temp, diagonal=1)  # Convert to mask and return


def self_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Performs 'Scaled Dot Product Attention' from the 'Attention Is All You Need' paper (https://arxiv.org/abs/1706.03762)."""
    d_k = keys.size(-1)  # number of tokens/key dimension

    # Scores: For each token, determine the attention focus against all other tokens
    scores = torch.matmul(queries, keys.transpose(-2, -1))
    scaled_scores = scores / math.sqrt(d_k)

    # Mask scores if required
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)  # -1e9: not infinity but large enough!

    print(f'Scaled scores: {scaled_scores.size()} \n{scaled_scores}\n')
    attn_weights = torch.softmax(scaled_scores, dim=-1)
    print(f'Attention weights: {attn_weights.size()} \n{attn_weights}\n')

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
        # Create respective query, key, value pairs (same input for each one)
        query = self.split_to_heads(self.fc_query(query))
        key = self.split_to_heads(self.fc_key(key))
        value = self.split_to_heads(self.fc_value(value))

        attn_out = self_attention(queries=query, keys=key, values=value, mask=mask)  # [batch_size, n_heads, seq_len, head_dim]
        self.attention = self.combine_heads(attn_out)
        return self.fc_out(self.attention)  # [batch_size, seq_len, embed_size]


if __name__ == "__main__":
    VOCAB_SIZE = 10
    EMBED_SIZE = 3
    MAX_SEQ_LENGTH = 3
    BATCH_SIZE = 1
    NUM_HEADS = 1

    # Generate random sample data
    sm_data = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LENGTH))  # 3x3 matrix
    print(f'Data size: {sm_data.size()}')
    print(f'Data:\n {sm_data}')
    print('---------------------------------\n')

    # Embed data
    embedding = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)
    embedded_data = embedding(sm_data)

    mask = generate_mask(embedded_data)  # Mask it
    mh_attn = MultiHeadedAttention(embed_size=EMBED_SIZE, n_heads=NUM_HEADS)  # Create attention instance

    print('Attention (No Mask) -')
    attn_out = mh_attn(embedded_data, embedded_data, embedded_data)
    print(f'Attention output: {attn_out.size()} \n{attn_out}')
    print('---------------------------------\n')

    print('Attention (Masked) -')
    attn_out_masked = mh_attn(embedded_data, embedded_data, embedded_data, mask=mask)
    print(f'Attention output: {attn_out_masked.size()} \n{attn_out_masked}')
