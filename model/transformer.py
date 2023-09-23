import copy

from model.attention import MultiHeadedAttention
from model.embed import WordEmbeddings
from model.encoding import AbsolutePositionalEncoding
from model.ffn import PositionWiseFFN
from model.mask import generate_mask
from model.normalize import ResidualAndLayerNorm

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

    :param embed_dim: (int) the token embedding dimensions
    :param attn_module: (torch.nn.Module) the attention module block
    :param ffn_module: (torch.nn.Module) the feed-forward network module block
    :param drop_prob: (float, optional) the neuron dropout probability. Defaults to 0.1
    :param epsilon: (float, optional) a value added to the normalization for numerical stability. Default: 1e-6
    :param device: (string, optional) name of the PyTorch CUDA device to connect to (if CUDA is available). Defaults to cpu
    """
    def __init__(self, embed_dim: int, attn_module: nn.Module, ffn_module: nn.Module, drop_prob: float = 0.1,
                 epsilon: float = 1e-6, device: str = "cpu") -> None:
        super().__init__()
        self.attn = attn_module.to(device)
        self.ffn = ffn_module.to(device)
        self.residuals = stack(ResidualAndLayerNorm(embed_dim, drop_prob, epsilon, device), 2)
        self.size = embed_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Passes the input and mask through the block."""
        x = self.residuals[0](x, lambda x: self.attn(query=x, key=x, value=x, mask=mask))
        return self.residuals[1](x, self.ffn)


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
    :param device: (string, optional) name of the PyTorch CUDA device to connect to (if CUDA is available). Defaults to cpu
    """
    def __init__(self, embed_dim: int, masked_attn_module: nn.Module, attn_module: nn.Module, ffn_module: nn.Module,
                 drop_prob: float = 0.1, epsilon: float = 1e-6, device: str = "cpu") -> None:
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


class Classifier(nn.Module):
    """
    A simple classifier for converting an Encoder-Decoder model output into probabilities.

    :param embed_dim: (int) the input embedding size
    :param vocab_size: (int) the size of the vocabulary
    """
    def __init__(self, embed_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input through the classifier. Returns a probability distribution."""
        return self.softmax(self.linear(x))


class Transformer(nn.Module):
    """
    A basic representation of the Transformer architecture defined in the 'Attention Is All You Need' paper (https://arxiv.org/abs/1706.03762).

    :param src_vocab_size: (int) the vocabulary size of the encoder input
    :param tgt_vocab_size: (int) the vocabulary size of the decoder input
    :param embed_dim: (int, optional) the embedding dimension size. Defaults to `512`
    :param n_heads: (int, optional) the number of Multi-Headed Attention heads in the Transformer. Defaults to `8`
    :param n_layers: (int, optional) the number of stacks of the Encoder and Decoder blocks. Defaults to `6`
    :param hidden_dim: (int, optional) the number of nodes in the hidden layers. Defaults to `2048`
    :param max_seq_len: (int, optional) the maximum length a single sequence can be. Defaults to `1500`
    :param drop_prob: (float, optional) the dropout probability rate. Defaults to `0.1`
    :param epsilon: (float, optional) a value added to the normalization for numerical stability. Defaults to `1e-6`
    :param device: (string, optional) name of the PyTorch CUDA device to connect to (if CUDA is available). Defaults to `cpu`
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_dim: int = 512, n_heads: int = 8, n_layers: int = 6,
                 hidden_dim: int = 2048, max_seq_len: int = 1500, drop_prob: float = 0.1, epsilon: float = 1e-6, device: str = 'cpu') -> None:
        super().__init__()
        c = copy.deepcopy

        # Init helper layers/blocks
        self.attn = MultiHeadedAttention(embed_dim, n_heads, drop_prob)
        self.ffn = PositionWiseFFN(embed_dim, hidden_dim, drop_prob)
        self.pos_encoding = AbsolutePositionalEncoding(embed_dim, drop_prob, max_seq_len, device=device)
        self.dropout = nn.Dropout(drop_prob)

        # Init embeddings
        self.encoder_embeds = WordEmbeddings(src_vocab_size, embed_dim, device=device)
        self.decoder_embeds = WordEmbeddings(tgt_vocab_size, embed_dim, device=device)

        # Set Transformer layers
        self.encoder_layers = stack(EncoderBlock(embed_dim, c(self.attn), c(self.ffn), drop_prob, epsilon, device), n_layers).to(device)
        self.decoder_layers = stack(DecoderBlock(embed_dim, c(self.attn), c(self.attn), c(self.ffn), drop_prob, epsilon, device), n_layers).to(device)
        self.classifier = Classifier(embed_dim, tgt_vocab_size).to(device)

        # Store the device and init params
        self.device = device
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize the parameters using Xavier's method."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Passes the source and target tensors through the Transformer."""
        # Compute masks and embeddings
        src_mask, tgt_mask = generate_mask(src), generate_mask(tgt)
        src_embeds = self.dropout(self.pos_encoding(self.encoder_embeds(src))).to(self.device)
        tgt_embeds = self.dropout(self.pos_encoding(self.encoder_embeds(tgt))).to(self.device)

        # Iterate over each encoder layer, return last ones output
        enc_output = src_embeds
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # Iterate over each decoder layer, return last ones output
        dec_output = tgt_embeds
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        # Pass final output to the classifier
        return self.classifier(dec_output)
