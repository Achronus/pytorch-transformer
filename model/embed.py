"""A simple example for creating vector embeddings for a small vocabulary using a custom Embedding Layer."""
import logging
from logger import enable_logging

import torch
import torch.nn as nn

logger = logging.getLogger('embeddings')


class WordEmbeddings(nn.Module):
    """
    A basic representation of a Word Embedding layer.

    :param vocab_size (int) - size of the vocabulary dictionary
    :param vec_size: (int) - number of embeddings per input item in a single vector
    """
    def __init__(self, vocab_size: int, vec_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.vec_size = vec_size

        self.embedding_weights = nn.Parameter(torch.Tensor(vocab_size, vec_size))
        self.reset_parameters()

    def __init_weights(self, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        """Normalize the embeddings with a mean of 0 and std of 1."""
        with torch.no_grad():
            return self.embedding_weights.normal_(mean, std)

    def reset_parameters(self):
        self.__init_weights()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Retrieve the embeddings (weights) associated to each unique input value. Each one acts as an
        index to a respective set of embedding weights (vector embedding)."""
        try:
            if x.dim() == 1:
                return self.embedding_weights[x]
            elif x.dim() == 2:
                return self.embedding_weights.index_select(0, x.view(-1))
            else:
                raise ValueError("Input must be 1-D or 2-D tensor.")
        except IndexError:
            raise IndexError(f"'embedding_size' must be > 'vocab_size'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create vector embeddings for a given input."""
        return self.embed(x)


class PatchEmbeddings(nn.Module):
    """
    Converts a batch of images into patches and projects them into a vector space.

    :param img_size: (int) the size of one dimension of the image (WxH must match)
    :param patch_size: (int)
    :param n_channels: (int) number of image colour channels
    :param n_embeds: (int) number of embeddings per patch (output filters)
    :param log_info: (bool, optional) a flag for enabling logging information. Defaults to False
    """
    def __init__(self, img_size: int, patch_size: int, n_channels: int, n_embeds: int,
                 log_info: bool = False) -> None:
        super().__init__()
        enable_logging(logger, flag=log_info)

        # Comments follow example: batch_size=1, img_size=32, patch_size=4, n_channels=3, n_embeds=5
        self.patch_dim = img_size // patch_size  # 32x32 -> 8
        self.pixels_per_patch = self.patch_dim ** 2  # 32x32 -> 8x8 patches = 64 pixels

        # Create learnable token - initialized with Gaussian distribution
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embeds))

        # Convert image into patches using a single layer
        self.conv = nn.Conv2d(n_channels, n_embeds, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create patch embeddings with a cls token prepended."""
        # x -> (batch_size, num_channels, img_size, img_size) = (1, 3, 32, 32)
        # x_conv -> (batch_size, n_embeds, patch_dim, patch_dim) = (1, 5, 8, 8)
        x_conv = self.conv(x)
        logger.debug(f'x_conv size: {x_conv.size()}')

        # x_final -> (batch_size, pixels_per_patch, n_embeds) = (1, 64, 5)
        x_final = x_conv.flatten(2).transpose(1, 2)
        logger.debug(f'x_final size: {x_final.size()}')

        # cls_token -> (batch_size, 1, n_embeds) = (1, 1, 5)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        logger.debug(f'cls_token size: {cls_token.size()}')

        # output -> (batch_size, pixels_per_patch + 1, n_embeds) = (1, 65, 5)
        return torch.cat([cls_token, x_final], dim=1)
