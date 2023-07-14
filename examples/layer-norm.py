from model.normalize import LayerNormalization

import torch
import torch.nn as nn


if __name__ == "__main__":
    NUM_EMBEDDINGS = 10
    EMBED_SIZE = 3

    torch.manual_seed(6)
    sample = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    print(f'Before: {sample.size()} \n{sample}')

    # Embed data
    embedding = nn.Embedding(NUM_EMBEDDINGS, EMBED_SIZE)
    embed_data = embedding(sample)
    embed_shape = embed_data.size()

    # Simple feed-forward network
    ff = nn.Sequential(
        nn.Linear(EMBED_SIZE, 100),
        nn.ReLU(),
        nn.Linear(100, EMBED_SIZE)
    )

    # Custom PyTorch module
    ln = LayerNormalization(embed_shape)
    x = ln(ff(embed_data))
    print(f'After LayerNorm (custom): {x.size()} \n{x}\n')

    # Compare against PyTorch built-in
    ln2 = nn.LayerNorm(embed_shape, eps=1e-6)
    x2 = ln2(ff(embed_data))
    print(f'After LayerNorm (PyTorch): {x2.size()} \n{x2}')
