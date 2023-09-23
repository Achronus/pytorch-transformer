# PyTorch Transformer
This repository focuses on a standard Transformer architecture based on the *["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)* paper created using the PyTorch library.

It accompanies a series of Transformer blog posts found on [Medium](https://medium.com/@achronus/transformers-the-frontier-of-ml-generalisation-9659c42111c9) 
that intend to provide a deep understanding of the architecture. You can view a roadmap of the repository below.

# Roadmap
We divide the roadmap into two sections: `components` and `demos`.

## Components
This section focuses on the components of the architecture that are progressively added to the repository. They are housed in the `/model` folder with 
respective `.py` files.
- [x] Self-Attention | `model/attention.py`
- [x] Multi-Headed Attention | `model/attention.py`
- [x] Mask Generation | `model/mask.py`
- [x] Embedding Methods | `model/embed.py`
- [x] Absolute (Sinusoidal) Positional Encoding | `model/encoding.py`
- [x] Position-Wise Feed-Forward Networks | `model/ffn.py`
- [x] Residual Connections | `model/normalize.py`
- [x] Layer Normalisation | `model/normalize.py`
- [x] Encoders | `model/transformer.py`
- [x] Decoders | `model/transformer.py`
- [x] Transformer | `model/transformer.py`

Additional components are found in the `/utils` folder with the respective `.py` files.
- [x] Raw Text File Reader | `utils/reader.py`
- [x] Word Tokenizer | `utils/tokenize.py`
- [x] Logger | `utils/logger.py`
- [x] Latent Semantic Analysis (LSA) | `utils/lsa.py`
- [x] Pairwise Inner Product (PIP) Loss | `utils/pip.py` 

## Demos
These are found in the `/examples` folder and consist of simple demos (small tutorials) for specific components to help with debugging the code (e.g., checking tensor dimensions) and act as a quickstart guide for them. You will find them in the format of `[name].py` for code only (with comments).

- [x] Attention | `examples/attention.py`
- [x] Masking Attention | `examples/attention_masking.py`
- [x] Layer Normalisation | `examples/layer_norm.py`
- [x] Word Embeddings | `examples/embed_vocab.py`
- [x] Image Embeddings | `examples/embed_imgs.py`
- [x] Finding Optimal Embedding Dimension | `examples/optimal_embed_dim.py`
- [x] Visualising Positional Encoding | `examples/pos_encoding.py`
- [x] Transformer Creation | `examples/transformer_demo.py`

# References
Huang, A., Subramanian, S., Sum, J., Almubarak, K., and Biderman, S., 2022. *The Annotated Transformer*. [online] Harvard University. Available from: 
http://nlp.seas.harvard.edu/annotated-transformer/.

Sarkar, A., 2023. *Build Your Own Transformer From Scratch Using PyTorch*. [online] Medium. Available from: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb.