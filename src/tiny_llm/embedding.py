import mlx.core as mx

from tiny_llm_ref.basics import linear



class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x,:]

    def as_linear(self, x: mx.array) -> mx.array:
        return linear(x, self.weight)
