import torch
import torch.nn as nn


def generate_mask(x: torch.Tensor, mask_value: int = -1e9) -> torch.Tensor:
    """Creates a mask containing 0s and a custom value on the upper diagonal. Mask is the same size as the given input."""
    # Create a temporary tensor that matches the input size
    temp = torch.ones(x.size())
    temp = torch.fill(temp, mask_value)  # Fill with custom value
    return torch.triu(temp, diagonal=1)  # Convert to mask and return


def mask_scores(x: torch.Tensor) -> torch.Tensor:
    """Generates a mask as the same size as the given input and automatically applies the mask to it."""
    mask = generate_mask(x)
    return torch.softmax(x + mask, dim=-1)


if __name__ == "__main__":
    VOCAB_SIZE = 10
    EMBED_SIZE = 3
    MAX_SEQ_LENGTH = 3
    BATCH_SIZE = 1

    # Generate random sample data
    sm_data = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LENGTH))  # 3x3 matrix

    # Embed it
    embedding = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)
    embedded_data = embedding(sm_data)

    # Mask it
    mask = generate_mask(embedded_data)
    masked_scores = mask_scores(embedded_data)

    # View outputs for data sizes and data format
    print(f'Data size: {sm_data.size()}')
    print(f'Data:\n {sm_data}\n')

    print(f'Embedded data size: {embedded_data.size()}')
    print(f'Embedded data:\n {embedded_data}\n')

    print(f'Mask size: {mask.size()}')
    print(f'Mask:\n {mask}\n')

    print(f'Masked scores size: {masked_scores.size()}')
    print(f'Masked scores:\n {masked_scores}')
