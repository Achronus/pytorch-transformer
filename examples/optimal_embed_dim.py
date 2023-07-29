import os

from model.embed import WordEmbeddings
from utils.tokenize import WordTokenizer
from utils.reader import RawTextReader
from utils.lsa import LSA
from utils.pip import PIPLoss

if __name__ == '__main__':
    vec_size = 5
    skip_window = 5

    reader = RawTextReader(filename=f'{os.getcwd()}/data/text8.gz')
    data = reader.read_data()

    tokenizer = WordTokenizer(data[:100000], log_info=True)
    test_data = tokenizer.indexed_corpus

    # Train Latent Semantic Analysis model
    lsa = LSA(test_data, skip_window, log_info=True)
    lsa_results = lsa.train()

    # Get optimal vector size using PIP Loss
    pip = PIPLoss(lsa_results, log_info=True)
    optimal_vec_size = pip.compute_optimal_dim()
    print(f'optimal_dim={optimal_vec_size}')
    pip.plot_losses()

    # Create word embeddings
    embedding = WordEmbeddings(len(tokenizer.vocab), optimal_vec_size)

    # Supply to desired Transformer
    # ...
