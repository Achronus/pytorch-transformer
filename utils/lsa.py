import collections
import logging
from dataclasses import dataclass

from logger import enable_logging

import torch


logger = logging.getLogger('lsa')


@dataclass
class LSAResults:
    """A data class for storing the LSA results."""
    spectrum: torch.Tensor
    noise: float
    alpha: float


class LSA:
    """
    An implementation of Latent Semantic Analysis (LSA) based on Yin and Shen's (2018) work
    in the paper `On The Dimensionality of Word Embedding`. Original code found here:
    https://github.com/ziyin-dl/word-embedding-dimensionality-selection.
    Paper found on arxiv here: https://arxiv.org/abs/1812.04224. Refer to section 2.2 for LSA.

    :param indexed_corpus: (torch.Tensor) a corpus of values indexed based on a vocabulary
    :param skip_window: (int, optional) a value for obtaining the co-occurrence of words in the corpus. Defaults to 5
    :param alpha: (float, optional) a hyperparameter to enforce symmetry between matrix factorization methods. Defaults to 0.5
    :param device: (string, optional) name of the PyTorch CUDA device to connect to (if CUDA is available). Defaults to cuda:0
    """

    def __init__(self, indexed_corpus: torch.Tensor, skip_window: int = 5, alpha: float = 0.5,
                 device: str = "cuda:0", log_info: bool = False) -> None:
        enable_logging(logger, flag=log_info)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.corpus = indexed_corpus
        self.skip_window = skip_window
        self.vocab_size = len(set(self.corpus.tolist()))  # Number of unique items in corpus
        self.alpha = alpha

        self.results = None

    def train(self) -> LSAResults:
        """Trains the model on the class provided corpus."""
        spectrum = self.calc_spectrum()
        noise = self.calc_noise()
        logger.debug(f'spectrum={spectrum.size()}, noise={noise}, alpha={self.alpha}')
        self.results = LSAResults(spectrum=spectrum, noise=noise, alpha=self.alpha)
        return self.results

    def calc_spectrum(self) -> torch.Tensor:
        """Computes the Singular Value Decomposition (SVD) of the Positive PMI.
        Returns only the spectrum (D)."""
        positive_pmi = self.calc_positive_pmi(self.corpus)  # M (signal_matrix)
        u, d, v = torch.linalg.svd(positive_pmi)
        return d  # spectrum

    def calc_noise(self) -> float:
        """Implements the 'count-twice' trick as mentioned in section 5.2.1 of the paper, to estimate
        noise."""
        # Calculate splits
        data_size = self.corpus.size(0)
        first_half_corpus = self.corpus[:data_size // 2]
        second_half_corpus = self.corpus[data_size // 2 + 1:]

        # Calculate Positive PMIs
        first_positive_pmi = self.calc_positive_pmi(first_half_corpus)
        second_positive_pmi = self.calc_positive_pmi(second_half_corpus)

        # Compute difference and calculate noise
        diff = first_positive_pmi - second_positive_pmi
        return torch.std(diff) * 0.5  # noise

    def __build_co_occurrence_dict(self, corpus: torch.Tensor) -> dict[int, collections.Counter]:
        """Creates a co-occurrence dictionary from a given corpus."""
        vocab_size = self.vocab_size
        co_occurrence_dict = collections.defaultdict(collections.Counter)
        corpus_size = corpus.size(0)

        # Add corpus indices (key) to dictionary with surrounding items (value)
        for idx, center_word_id in enumerate(corpus):
            # Reset vocab_size (loop round when reach maximum)
            if center_word_id > vocab_size:
                vocab_size = center_word_id

            # Get co-occurrence values (around each word) based on skip window
            val = center_word_id.item()  # unpack from tensor
            high = max(idx - self.skip_window - 1, 0)
            low = min(idx + self.skip_window + 1, corpus_size)

            # Set co-occurrence counts to 1, if same value 0
            for i in range(high, low):
                co_occurrence_dict[val][corpus[i].item()] += 1
            co_occurrence_dict[val][val] -= 1

        # E.g., defaultdict({136: Counter({221: 1, 12: 1, 488: 1, 5: 1, 6: 1, 136: 0}, ...)

        logger.debug(f"co_occurrence_dict={len(co_occurrence_dict)}")
        return co_occurrence_dict

    def __create_doc_term_matrix(self, corpus: torch.Tensor) -> torch.Tensor:
        """Creates a document term matrix using a given corpus."""
        occur_dict = self.__build_co_occurrence_dict(corpus)  # Get co-occurrence dict

        # term_matrix: col_pos=token_idx (dim=0), row_pos=neighbour_idx_count (dim=1)
        term_matrix = torch.zeros([self.vocab_size, self.vocab_size])

        # Iterate over occur_dict
        for i, counter in occur_dict.items():
            for j, count in counter.items():
                # Set neighbour_idx_counts to rows for each token_idx
                term_matrix[i, j] += count

        logger.debug(f"term_matrix={term_matrix.size()}")
        return term_matrix

    def calc_positive_pmi(self, corpus: torch.Tensor) -> torch.Tensor:
        """Computes the Positive Pointwise Mutual Information (PMI) using a given corpus."""
        term_matrix = self.__create_doc_term_matrix(corpus).to(self.device)

        # Compute probabilities
        vocab_counts = term_matrix.sum(dim=1)  # Counts per token_idx
        total_count = term_matrix.sum()  # Total vocab counts
        term_matrix_probs = term_matrix / total_count
        vocab_counts_probs = vocab_counts / torch.sum(vocab_counts)
        logger.debug(f'vocab={vocab_counts.size()}, total={total_count}')

        # Calculate Pointwise Mutual Information (PMI)
        vocab_probs_outer = torch.outer(vocab_counts_probs, vocab_counts_probs)
        pmi = torch.log(term_matrix_probs) - torch.log(vocab_probs_outer)

        # Tidy matrix removing -inf and nan
        pmi[torch.isinf(pmi)] = 0
        pmi[torch.isnan(pmi)] = 0

        # Calculate positive PMI
        positive_pmi = pmi
        positive_pmi[positive_pmi < 0] = 0
        logger.debug(f'pmi={pmi.size()}, positive_pmi={positive_pmi.size()}\n')
        return positive_pmi
