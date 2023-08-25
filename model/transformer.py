import copy

from normalize import LayerNormalization, ResidualAndLayerNorm

import torch
import torch.nn as nn


def stack(module: torch.nn, N: int) -> nn.ModuleList:
    """
    Creates a stack of N identical blocks.

    :param module: (torch.nn.Module) the PyTorch module to duplicate
    :param N: (int) the number of layers to create
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderBlock(nn.Module):
    """
    An Encoder block found in the Transformer architecture defined in the
    'Attention Is All You Need' paper (https://arxiv.org/abs/1706.03762). Can be stacked multiple times.

    :param embed_dim: (tuple[int, ...] | torch.size) the token embedding dimensions
    :param attn_module: (torch.nn.Module) the attention module block
    :param ffn_module: (torch.nn.Module) the feed-forward network module block
    :param drop_prob: (float, optional) the neuron dropout probability. Defaults to 0.1
    :param epsilon: (float, optional) a value added to the normalization for numerical stability. Default: 1e-6
    :param device: (string, optional) name of the PyTorch CUDA device to connect to (if CUDA is available). Defaults to cuda:0
    """
    def __init__(self, embed_dim: tuple[int, ...], attn_module: nn.Module, ffn_module: nn.Module, drop_prob: float = 0.1,
                 epsilon: float = 1e-6, device: str = "cuda:0") -> None:
        super().__init__()
        self.attn = attn_module.to(device)
        self.ffn = ffn_module.to(device)
        self.residuals = stack(ResidualAndLayerNorm(embed_dim, drop_prob, epsilon, device), 2)
        self.size = embed_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Passes the input and mask through the block."""
        x = self.sublayer[0](x, lambda x: self.attn(query=x, key=x, value=x, mask=mask))
        return self.sublayer[1](x, self.ffn)


class DecoderBlock(nn.Module):
    """
    A Decoder block found in the Transformer architecture defined in the
    'Attention Is All You Need' paper (https://arxiv.org/abs/1706.03762). Can be stacked multiple times.

    :param embed_dim: (tuple[int, ...] | torch.size) the token embedding dimensions
    :param masked_attn_module: (torch.nn.Module) the masked attention module block
    :param attn_module: (torch.nn.Module) the attention module block
    :param ffn_module: (torch.nn.Module) the feed-forward network module block
    :param drop_prob: (float, optional) the neuron dropout probability. Defaults to 0.1
    :param epsilon: (float, optional) a value added to the normalization for numerical stability. Default: 1e-6
    :param device: (string, optional) name of the PyTorch CUDA device to connect to (if CUDA is available). Defaults to cuda:0
    """
    def __init__(self, embed_dim: tuple[int, ...], masked_attn_module: nn.Module, attn_module: nn.Module, ffn_module: nn.Module,
                 drop_prob: float = 0.1, epsilon: float = 1e-6, device: str = "cuda:0") -> None:
        super().__init__()
        self.masked_attn = masked_attn_module.to(device)
        self.attn = attn_module.to(device)
        self.ffn = ffn_module.to(device)
        self.residuals = stack(ResidualAndLayerNorm(embed_dim, drop_prob, epsilon, device), 3)
        self.size = embed_dim

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """Passes the input, encoder output, and masks through the block."""
        e_out = encoder_out
        x = self.residuals[0](x, lambda x: self.masked_attn(query=x, key=x, value=x, mask=tgt_mask))
        x = self.residuals[1](x, lambda x: self.attn(query=x, key=e_out, value=e_out, mask=src_mask))
        return self.residuals[2](x, self.ffn)


class Encoder(nn.Module):
    """
    A basic representation of multiple Encoder blocks.

    :param layer: (EncoderBlock) the type of EncoderBlock to use
    :param N: (int) the number of EncoderBlocks to stack together
    :param epsilon: (float, optional) a value added to the normalization for numerical stability. Default: 1e-6
    :param device: (string, optional) name of the PyTorch CUDA device to connect to (if CUDA is available). Defaults to cuda:0
    """
    def __init__(self, layer: EncoderBlock, N: int, epsilon: float = 1e-6, device: str = "cuda:0") -> None:
        super().__init__()
        self.layers = stack(layer, N)
        self.norm = LayerNormalization(layer.size, epsilon, device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Passes the input and mask through each layer."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    A basic representation of multiple Decoder blocks.

    :param layer: (DecoderBlock) the type of EncoderBlock to use
    :param N: (int) the number of EncoderBlocks to stack together
    :param epsilon: (float, optional) a value added to the normalization for numerical stability. Default: 1e-6
    :param device: (string, optional) name of the PyTorch CUDA device to connect to (if CUDA is available). Defaults to cuda:0
    """
    def __init__(self, layer: DecoderBlock, N: int, epsilon: float = 1e-6, device: str = "cuda:0") -> None:
        super().__init__()
        self.layers = stack(layer, N)
        self.norm = LayerNormalization(layer.size, epsilon, device)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """Passes the input, encoder output and masks through each layer."""
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        return self.norm(x)
