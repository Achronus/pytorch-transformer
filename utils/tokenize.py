from utils.logger import create_logger

import torch
from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator


class WordTokenizer:
    """
    A simple word tokenizer for extracting words and indices from a corpus.

    :param text: (str) a corpus of text data as a string
    :param device: (string, optional) name of the PyTorch CUDA device to connect to (if CUDA is available). Defaults to cuda:0
    :param log_info: (bool, optional) a flag for enabling logging information. Defaults to False
    """
    def __init__(self, text: str, device: str = "cuda:0", log_info: bool = False) -> None:
        self.logger = create_logger(self.__class__.__name__, filename='tokenizer', flag=log_info)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.data = text
        self.tokenizer = get_tokenizer('basic_english')
        self.tokenized_corpus = self.__tokenize_words()  # corpus as list of words
        self.vocab = self.__build_vocab().to(self.device)
        self.vocab_dict = self.vocab.get_stoi()
        self.indexed_corpus = self.tokens_to_indices(self.tokenized_corpus).to(self.device)

    def __tokenize_words(self) -> list[str]:
        """Converts a corpus of text into a list of words and punctuation."""
        words = self.tokenizer(self.data)
        self.logger.info(f'Words in corpus: {len(words)}')
        return words

    def __build_vocab(self) -> Vocab:
        """Creates a vocabulary from the tokenized words, taking their unique values and allocating an index to them."""
        vocab = build_vocab_from_iterator([self.tokenized_corpus])
        self.logger.info(f'Words in vocabulary: {len(vocab)}')
        return vocab

    def tokens_to_indices(self, tokens: list[str]) -> torch.Tensor:
        """Retrieves a set of tokens indices."""
        return torch.LongTensor(self.vocab.lookup_indices(tokens))
