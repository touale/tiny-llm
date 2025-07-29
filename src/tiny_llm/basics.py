import mlx.core as mx
import mlx.nn as nn


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    out = x @ w.T
    if bias is not None:
        out += bias
    return out



def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)
