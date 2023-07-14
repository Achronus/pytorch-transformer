from model.embed import WordEmbeddings

import torch


if __name__ == '__main__':
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
