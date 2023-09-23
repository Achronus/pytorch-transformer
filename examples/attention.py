"""
A code example demo of Multi-Headed Attention. Uses the same code from the `/model` directory with the addition of print statements for tensor debugging.
Additionally, contains an example of PositionalEncoding and torch.nn.Embedding to demonstrate a full implementation.
"""

import math
from model.attention import MultiHeadedAttention
from model.encoding import AbsolutePositionalEncoding

import torch
import torch.nn as nn


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
    pos_encode = AbsolutePositionalEncoding(embed_dim=EMBED_SIZE, drop_prob=0.1, max_len=MAX_SEQ_LENGTH, log_info=True)
    embedding = nn.Embedding(SRC_VOCAB_SIZE, EMBED_SIZE)

    encoded_input = pos_encode(embedding(src_data))
    out = mh_attn.forward(encoded_input, encoded_input, encoded_input)  # [64, 100, 512] -> [batch_size, seq_len, embed_size]
    print(f'Final output: {out.size()}')

