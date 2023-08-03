import os

from model.embed import WordEmbeddings
from utils.tokenize import WordTokenizer
from utils.reader import RawTextReader
from utils.lsa import LSA
from utils.pip import PIPLoss

if __name__ == '__main__':
    skip_window = 5

    reader = RawTextReader(filename=f'{os.getcwd()}/data/text8.gz')
    data = reader.read_data()

    print('Tokenizing words...', end='')
    tokenizer = WordTokenizer(data[:100000], log_info=True)
    test_data = tokenizer.indexed_corpus
    print('Complete.')

    # Train Latent Semantic Analysis model
    print('Training LSA...', end='')
    lsa = LSA(test_data, skip_window, log_info=True)
    lsa_results = lsa.train()
    print('Complete.')

    # Get optimal vector size using PIP Loss
    print('Computing optimal dimension...', end='')
    pip = PIPLoss(lsa_results, log_info=True)
    optimal_vec_size = pip.compute_optimal_dim()
    print('Complete.')
    print(f'optimal_dim={optimal_vec_size}')
    pip.plot_losses()

    # Create word embeddings
    embedding = WordEmbeddings(len(tokenizer.vocab), optimal_vec_size)

    # Supply to desired Transformer
    # ...
