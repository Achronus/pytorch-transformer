import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    A basic representation of an Embedding layer.

    :param embedding_size (int) - size of the vocabulary dictionary
    :param vec_size: (int) - number of embeddings per input item in a single vector
    """
    def __init__(self, embedding_size: int, vec_size: int) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.vec_size = vec_size

        self.embedding_weights = nn.Parameter(torch.Tensor(embedding_size, vec_size))
        self.reset_parameters()

    def __init_weights(self, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        """Normalize the embeddings with a mean of 0 and std of 1."""
        with torch.no_grad():
            return self.embedding_weights.normal_(mean, std)

    def reset_parameters(self):
        self.__init_weights()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Retrieve the embeddings (weights) associated to each input value. Each input value acts as an
        index to a respective set of embedding weights (vector embedding)."""
        try:
            if x.dim() == 1:
                return self.embedding_weights[x]
            elif x.dim() == 2:
                return self.embedding_weights.index_select(0, x.view(-1))
            else:
                raise ValueError("Input must be 1-D or 2-D tensor.")
        except IndexError:
            raise IndexError(f"'embedding_size' must be greater than 'x.view(-1)'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create vector embeddings for a given input."""
        return self.embed(x)


if __name__ == '__main__':
    embedding_size = 10
    vec_size = 5

    embedding = EmbeddingLayer(embedding_size, vec_size)

    # An input tensor of values (acting as indices)
    vocab = {"a": 0, "this": 1, "is": 2, "sentence": 3}
    input_tensor = torch.LongTensor(list(vocab.values()))

    # Forward pass
    output = embedding(input_tensor)

    print(output.size(), '\n', output)
