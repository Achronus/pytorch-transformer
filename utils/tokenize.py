import logging
from logger import enable_logging

import torch
from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator


logger = logging.getLogger('tokenizer')


class WordTokenizer:
    """
    A simple word tokenizer for extracting words and indices from a corpus.

    :param text: (str) a corpus of text data as a string
    :param log_info: (bool, optional) a flag for enabling logging information. Defaults to False
    """
    def __init__(self, text: str, log_info: bool = False) -> None:
        enable_logging(logger, flag=log_info)

        self.data = text
        self.tokenizer = get_tokenizer('basic_english')
        self.tokenized_corpus = self.__tokenize_words()
        self.vocab = self.__build_vocab()
        self.vocab_dict = self.vocab.get_stoi()
        self.indexed_corpus = torch.LongTensor(list(self.vocab_dict.values()))

    def __tokenize_words(self) -> list[str]:
        """Converts a corpus of text into a list of words and punctuation."""
        words = self.tokenizer(self.data)
        logger.debug(f'Words in corpus: {len(words)}')
        return words

    def __build_vocab(self) -> Vocab:
        """Creates a vocabulary from the tokenized words, taking their unique values and allocating an index to them."""
        vocab = build_vocab_from_iterator([self.tokenized_corpus])
        logger.debug(f'Words in vocabulary: {len(vocab)}')
        return vocab
