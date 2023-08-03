from model.attention import MultiHeadedAttention
from model.mask import generate_mask
from utils.logger import create_logger

import torch
import torch.nn as nn

logger = create_logger('masking', filename='masking', flag=True)


if __name__ == "__main__":
    VOCAB_SIZE = 10
    EMBED_SIZE = 3
    NUM_HEADS = 1
    MAX_SEQ_LENGTH = 3
    BATCH_SIZE = 1

    # Generate random sample data
    sm_data = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LENGTH))  # 1x3 matrix
    logger.info(f'Data size: {sm_data.size()}')
    logger.info(f'Data: {sm_data}\n')

    # Embed data
    embedding = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)
    embedded_data = embedding(sm_data)

    mask = generate_mask(embedded_data)  # Mask it
    mh_attn = MultiHeadedAttention(embed_size=EMBED_SIZE, n_heads=NUM_HEADS, log_info=True)  # Create attention instance

    logger.info('Attention (No Mask) -')
    attn_out = mh_attn(embedded_data, embedded_data, embedded_data)
    logger.info(f'Attention size: {attn_out.size()}')
    logger.info(f'Attention data: \n\t{attn_out}\n')

    logger.info('Attention (Masked) -')
    attn_out_masked = mh_attn(embedded_data, embedded_data, embedded_data, mask=mask)
    logger.info(f'Attention size: {attn_out_masked.size()}')
    logger.info(f'Attention data: \n\t{attn_out_masked}')
