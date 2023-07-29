import math
import os
import logging

from logger import enable_logging
from utils.lsa import LSAResults

import torch
import matplotlib.pyplot as plt

logger = logging.getLogger('lsa')


class PIPLoss:
    """
    An implementation of the Pairwise Inner Product (PIP) Loss based on Yin and Shen's (2018) work
    in the paper `On The Dimensionality of Word Embedding`. Original code found here:
    https://github.com/ziyin-dl/word-embedding-dimensionality-selection.
    Paper found on arxiv here: https://arxiv.org/abs/1812.04224. Refer to section 3 for the PIP Loss.

    :param lsa_results: (LSAResults) results computed by an LSA model (spectrum, noise, and alpha)
    :param device: (string) name of the PyTorch CUDA device to connect to (if CUDA is available). Defaults to cuda:0
    """

    def __init__(self, lsa_results: LSAResults, device: str = "cuda:0", log_info: bool = False) -> None:
        enable_logging(logger, flag=log_info)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.spectrum = lsa_results.spectrum.to(self.device)  # lambda
        self.sigma = lsa_results.noise
        self.alpha = lsa_results.alpha
        self.result_folder = f'{os.getcwd()}/graphs'

        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)

        self.losses = None

    @staticmethod
    def __create_orthogonal_matrix(shape: tuple[int, int]) -> torch.Tensor:
        """Helper function for creating a random orthogonal matrix."""
        x = torch.normal(0, 1, size=shape)
        u, _, _, = torch.linalg.svd(x, full_matrices=False)
        return u

    def create_signal_and_rank(self) -> tuple[torch.Tensor, int]:
        """Creates an estimated signal matrix and assigns the value after the last non-zero signal (rank).
        Returns them both."""
        # Compute Universal Singular Value Threshold (USVT)
        tau = 2 * self.sigma * math.sqrt(self.spectrum.size(0))  # 0 signal threshold

        # Compute signal matrix
        signal_matrix = torch.where(self.spectrum > tau, self.spectrum - tau, 0.)  # (condition, true, false)

        # Set rank: idx of first 0 item
        rank: int = (signal_matrix == 0).nonzero()[0].item()
        return signal_matrix, rank

    def compute_optimal_dim(self) -> int:
        """Computes the optimal embedding dimension based on the supplied models results."""
        signal_matrix, rank = self.create_signal_and_rank()
        n = signal_matrix.size(0)
        shape = (n, n)
        logger.debug(f'n={n}, rank={rank}, sigma={self.sigma}')

        signals = signal_matrix[:rank].to(self.device)  # non-zero (d matrix)
        u_matrix = self.__create_orthogonal_matrix((n, rank)).to(self.device)
        v_matrix = self.__create_orthogonal_matrix((n, rank)).to(self.device)
        valid_signal_dims = range(rank)
        logger.debug(f'd_matrix={signals.size()}, u_matrix={u_matrix.size()}, v_matrix={v_matrix.size()}')

        # Compute x and y (target)
        x = (u_matrix * signals).matmul(v_matrix.T)  # (vocab_size, vocab_size)

        embed_matrix = torch.normal(0, self.sigma, size=shape).to(self.device)
        target = x + embed_matrix
        logger.debug(f'x={x.size()}, target={target.size()}')

        # Compute x and y SVD components
        u, d, v = torch.linalg.svd(x)
        u1, d1, v1 = torch.linalg.svd(target)

        # Compute oracle (true) and estimate embeddings
        embed_oracle = u[:, valid_signal_dims] * d[valid_signal_dims].pow(self.alpha)
        spectrum = d.pow(self.alpha)
        spectrum_estimate = d1.pow(self.alpha)
        embed_estimate = u1 * spectrum_estimate
        logger.debug(f'oracle_embed={embed_oracle.size()}, estimate_embed={embed_estimate.size()}')

        # Compute Frobenius norms
        pip_losses = [torch.linalg.norm(spectrum.pow(2)).pow(2)]  # Frobenius norms

        # Find best PIP loss
        for keep_dim in range(1, rank + 1):
            diff = pip_losses[keep_dim - 1] + \
                   spectrum_estimate[keep_dim - 1].pow(4) - 2 * \
                   (torch.linalg.norm(
                       self.__transpose_handler(
                           embed_estimate[:, keep_dim - 1]
                       ).matmul(embed_oracle)
                   ).pow(2))
            pip_losses.append(diff)

        self.losses = torch.Tensor(pip_losses[1:]).sqrt().cpu()
        return torch.argmin(self.losses).item()

    @staticmethod
    def __transpose_handler(tensor: torch.Tensor) -> torch.Tensor:
        """Ignores transpose if 1D tensor."""
        if len(tensor.size()) > 1:
            return tensor.T
        return tensor

    def plot_losses(self) -> None:
        """Generates a plot of stored pip losses."""
        if self.losses is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.losses, 'aqua', label=r'PIP Loss')
            legend = ax.legend(loc='upper right')
            plt.title(r'PIP Loss')
            fig_path = f'{self.result_folder}/pip_alpha{self.alpha}.png'
            fig.savefig(fig_path, bbox_extra_artists=(legend,), bbox_inches='tight')
            print(f'Plot loss diagram saved at {fig_path}')
            plt.show()
        else:
            print("No losses have been stored. Have you called '.compute_optimal_dim()'?")
