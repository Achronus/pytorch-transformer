"""
A code example demo of Multi-Headed Attention. Uses the same code from the `/model` directory with the addition of print statements for tensor debugging.
Additionally, contains an example of PositionalEncoding and torch.nn.Embedding to demonstrate a full implementation.
"""

import math
from model.attention import MultiHeadedAttention

import torch
import torch.nn as nn


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
    NUM_HEADS = 4
    MAX_SEQ_LENGTH = 5
    BATCH_SIZE = 2

    # Generate random sample data
    src_data = torch.randint(1, SRC_VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LENGTH))
    tgt_data = torch.randint(1, TGT_VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LENGTH))

    mh_attn = MultiHeadedAttention(embed_size=EMBED_SIZE, n_heads=NUM_HEADS, log_info=True)
    pos_encode = PositionalEncoding(embed_size=EMBED_SIZE, max_seq_length=MAX_SEQ_LENGTH)  # To be refactored/replaced
    embedding = nn.Embedding(SRC_VOCAB_SIZE, EMBED_SIZE)

    encoded_input = pos_encode(embedding(src_data))
    out = mh_attn.forward(encoded_input, encoded_input, encoded_input)  # [64, 100, 512] -> [batch_size, seq_len, embed_size]
    print(f'Final output: {out.size()}')

