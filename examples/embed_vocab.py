from model.embed import WordEmbeddings
from utils.tokenize import WordTokenizer
from utils.reader import RawTextReader

import torch


if __name__ == '__main__':
    # Simple Example
    embedding_size = 10
    vec_size = 5

    embedding = WordEmbeddings(embedding_size, vec_size)

    # An input tensor of values (acting as indices)
    vocab = {"a": 0, "this": 1, "is": 2, "sentence": 3}
    input_tensor = torch.LongTensor(list(vocab.values()))

    # Forward pass
    output = embedding(input_tensor)

    print(f'Vector Embeddings: {output.size()} \n{output}\n')
    print(f'Embedding Weights: {embedding.embedding_weights.size()}\n'
          f'{embedding.embedding_weights}')

    # More advanced example
    print('\n--------------')
    print('Running text8 example...')
    reader = RawTextReader(filename='../data/text8.gz')
    data = reader.read_data()

    tokenizer = WordTokenizer(data, log_info=True)
    test_data = tokenizer.indexed_corpus

    embedding2 = WordEmbeddings(len(tokenizer.vocab), vec_size)

    print(f'Text8 embedding size: {embedding2(test_data).size()}')
