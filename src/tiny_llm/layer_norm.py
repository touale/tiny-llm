import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def _norm(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.mean(x**2, axis=-1, keepdims=True) + self.eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self._norm(x.astype(mx.float32)).astype(x.dtype) * self.weight
