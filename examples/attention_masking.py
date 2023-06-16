import torch
import torch.nn as nn


def generate_mask(x: torch.Tensor) -> torch.Tensor:
    """Creates a mask containing 0s and 1s, where 0s are on the upper diagonal. Mask is the same size as the given input."""
    # Create a temporary tensor that matches the input size
    temp = torch.ones(x.size())
    return 1 - torch.triu(temp, diagonal=1)  # Convert to mask and return


def mask_scores(x: torch.Tensor) -> torch.Tensor:
    """Generates a mask as the same size as the given input and automatically applies the mask to it."""
    mask = generate_mask(x)
    print(f'Embedded data before masking:\n {x}\n')
    x = x.masked_fill(mask == 0, -1e9)  # -1e9: not infinity but large enough!
    print(f'Embedded data after masking:\n {x}\n')
    return torch.softmax(x + mask, dim=-1)


if __name__ == "__main__":
    VOCAB_SIZE = 10
    EMBED_SIZE = 3
    MAX_SEQ_LENGTH = 3
    BATCH_SIZE = 1

    # Generate random sample data
    sm_data = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LENGTH))  # 3x3 matrix
    print(f'Data size: {sm_data.size()}')
    print(f'Data:\n {sm_data}\n')

    # Embed it
    embedding = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)
    embedded_data = embedding(sm_data)

    # Mask it
    mask = generate_mask(embedded_data)
    masked_scores = mask_scores(embedded_data)

    # View outputs for data sizes and data format
    print(f'Mask size: {mask.size()}')
    print(f'Mask:\n {mask}\n')

    print(f'Masked scores size: {masked_scores.size()}')
    print(f'Masked scores:\n {masked_scores}')
