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
- [ ] Mask Generation
- [ ] Positional Encoding
- [ ] Position-Wise Feed-Forward Networks
- [ ] Residual Connections
- [ ] Layer Normalisation
- [ ] Encoders
- [ ] Decoders
- [ ] Transformer

## Demos
These are found in the `/examples` folder and are graded into two categories: `simple` and `advanced`.

`Simple` demos are small tutorials for specific components to help with debugging the code (e.g., checking tensor dimensions) and act as a quickstart 
guide for them. You will find them in the format of `[name].py` for code only (with comments) and `[name].ipynb` for a Jupyter Notebook variant with more 
detailed instructions.

Likewise, `advanced` items use the architecture in full-fledged projects. Typically, these are hosted in separate repositories that are linked accordingly.

- [ ] (Simple) Attention | `examples/attention.py`
- [ ] (Simple) Transformer Creation
- [ ] (Advanced) English-French Translation

# References
Huang, A., Subramanian, S., Sum, J., Almubarak, K., and Biderman, S., 2022. *The Annotated Transformer*. [online] Harvard University. Available from: 
http://nlp.seas.harvard.edu/annotated-transformer/.

Sarkar, A., 2023. *Build Your Own Transformer From Scratch Using PyTorch*. [online] Medium. Available from: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb.