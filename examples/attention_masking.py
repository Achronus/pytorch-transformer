import logging
from model.attention import MultiHeadedAttention

import torch
import torch.nn as nn

logger = logging.getLogger('masking')


def generate_mask(x: torch.Tensor) -> torch.Tensor:
    """Creates a mask containing 0s and 1s, where 0s are on the upper diagonal. Mask is the same size as the given input."""
    # Create a temporary tensor that matches the input size
    temp = torch.ones(x.size())
    return 1 - torch.triu(temp, diagonal=1)  # Convert to mask and return


if __name__ == "__main__":
    VOCAB_SIZE = 10
    EMBED_SIZE = 3
    MAX_SEQ_LENGTH = 3
    BATCH_SIZE = 1
    NUM_HEADS = 1

    # Generate random sample data
    sm_data = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LENGTH))  # 1x3 matrix
    logger.info(f'Data size: {sm_data.size()}')
    logger.info(f'Data: {sm_data}\n')

    # Embed data
    embedding = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)
    embedded_data = embedding(sm_data)

    mask = generate_mask(embedded_data)  # Mask it
    mh_attn = MultiHeadedAttention(embed_size=EMBED_SIZE, n_heads=NUM_HEADS, log_info=True)  # Create attention instance

    logger.info('---------------------------------')
    logger.info('Attention (No Mask) -')
    logger.info('---------------------------------')
    attn_out = mh_attn(embedded_data, embedded_data, embedded_data)
    logger.info(f'Attention size: {attn_out.size()}')
    logger.info(f'Attention data: \n\t{attn_out}\n')

    logger.info('---------------------------------')
    logger.info('Attention (Masked) -')
    logger.info('---------------------------------')
    attn_out_masked = mh_attn(embedded_data, embedded_data, embedded_data, mask=mask)
    logger.info(f'Attention size: {attn_out_masked.size()}')
    logger.info(f'Attention data: \n\t{attn_out_masked}')
