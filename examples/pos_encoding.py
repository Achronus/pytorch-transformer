import pandas as pd
import matplotlib.pyplot as plt

from model.embed import WordEmbeddings
from model.encoding import AbsolutePositionalEncoding

import torch


def create_data(data: torch.Tensor, n_positions: int, dims: list[int]) -> pd.DataFrame:
    """
    A helper function for creating the wave data to plot. Stored in a pd.DataFrame.

    :param data: (torch.Tensor) vector of positional encodings
    :param n_positions: (int) the vector size of the positional encodings
    :param dims: (list[int]) positional encoding dimensions to visualise
    """
    return pd.concat([
        pd.DataFrame(
            {
                "embedding": data[0, :, dim],
                "dimension": dim,
                "position": list(range(n_positions))
            }
        ) for dim in dims
    ])


def create_plot(y: torch.Tensor, sin_data: pd.DataFrame, cos_data: pd.DataFrame,
                sin_dims: list[int], cos_dims: list[int]) -> None:
    """
    A helper function to create a simple set of plots for visualising the sine and cosine embedding waves.

    :param y: (torch.Tensor) vector of positional encodings
    :param sin_data: (pd.DataFrame) plottable data of y following only sine dimensions
    :param cos_data: (pd.DataFrame) plottable data of y following only cosine dimensions
    :param sin_dims: (list[int]) a list of sine dimensions to plot
    :param cos_dims: (list[int]) a list of cosine dimensions to plot
    """
    fig, axs = plt.subplots(len(sin_dims), figsize=(12, 8))

    # Iterate over each plot
    for idx, dim in enumerate(sin_dims):
        # Add sine wave dims
        dim_data = sin_data[sin_data["dimension"] == dim]
        axs[idx].plot(dim_data["position"], dim_data["embedding"], color=f'C{idx}', label=f"{dim} (sine)")

        # Add cosine wave dims
        dim_data = cos_data[cos_data["dimension"] == cos_dims[idx]]
        axs[idx].plot(dim_data["position"], dim_data["embedding"], color=f'C{idx + len(cos_dims)}', label=f"{cos_dims[idx]} (cosine)")

        # Add embedding dots to plots
        for pos in DOT_POSITIONS:
            dot = y[0, pos, dim]
            dot2 = y[0, pos, cos_dims[idx]]

            # Add dots to plots
            axs[idx].plot(pos, dot, 'o', color='#730f9a')  # sine
            axs[idx].plot(pos, dot2, 'o', color='#730f9a')  # cosine

            # Add text to dots + line between dimensions
            axs[idx].text(pos + 0.4, dot - 0.15, f'p{pos}')
            axs[idx].plot([pos, pos], [dot, dot2], color='#730f9a', linewidth='1.5', linestyle='--')

        axs[idx].grid(visible=True)

    # Set axis labels and legend
    fig.supxlabel("Position", fontweight='bold')
    fig.supylabel("Embedding", fontweight='bold')
    fig.legend(title='Dimension', loc='upper right')
    plt.show()


if __name__ == '__main__':
    # PLOTTING EXAMPLE
    N_POSITIONS = 50
    EMBED_DIM = 20  # vec_size
    SIN_DIMS = [2, 4, 6]  # Sine
    COS_DIMS = [3, 5, 7]  # cosine
    DOT_POSITIONS = [2, 7, 16, 43]

    # Create dummy target values
    dummy_tensor = torch.zeros(1, N_POSITIONS, EMBED_DIM)
    pe = AbsolutePositionalEncoding(EMBED_DIM, drop_prob=0)
    y = pe.forward(dummy_tensor)

    # Create sine and cosine data
    sin_data = create_data(y, N_POSITIONS, SIN_DIMS)
    cos_data = create_data(y, N_POSITIONS, COS_DIMS)

    # Create a line plot for each dimension
    create_plot(y, sin_data, cos_data, SIN_DIMS, COS_DIMS)

    # ------------------------
    # MORE ADVANCED EXAMPLE
    # ------------------------
    VOCAB_SIZE = 5000
    EMBED_DIM = 512
    BATCH_SIZE = 64
    SEQ_LEN = 1500

    input_tensor = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    # Create word embeddings
    embedding = WordEmbeddings(VOCAB_SIZE, EMBED_DIM)
    x = embedding.forward(input_tensor.to('cuda:0'))

    # Init positional encoding
    pe = AbsolutePositionalEncoding(EMBED_DIM, drop_prob=0, log_info=True)
    y = pe.forward(x.cpu())
    print(x.size())
    print(y.size())
