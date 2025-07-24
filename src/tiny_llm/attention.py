import mlx.core as mx
from .basics import softmax, linear
import math


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    """
    attn = softmax((Q @ Kᵀ) / √dₖ + M) @ V

    """
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale

    scores = query @ key.swapaxes(-2, -1) * scale_factor

    if mask is not None:
        scores += mask

    p_attn = mx.softmax(scores, axis=-1)

    return p_attn @ value


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        """
        wq, wk, wv, wo: (BATCH_SIZE, L, H * D)
        """
        assert hidden_size % num_heads == 0, (
            "hidden_size must be divisible by num_heads"
        )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        bsz, seqlen, _ = query.shape

        # (BATCH_SIZE, L, H * D) -> (BATCH_SIZE, L, H × D)
        xq, xk, xv = (
            linear(query, self.wq),
            linear(key, self.wk),
            linear(value, self.wv),
        )

        # (BATCH_SIZE, L, H × D) -> (BATCH_SIZE, L, H, D)
        xq = mx.reshape(xq, (bsz, seqlen, self.num_heads, self.head_dim))
        xk = mx.reshape(xk, (bsz, seqlen, self.num_heads, self.head_dim))
        xv = mx.reshape(xv, (bsz, seqlen, self.num_heads, self.head_dim))

        # (BATCH_SIZE, L, H, D) -> (BATCH_SIZE, H, L, D)
        xq = mx.transpose(xq, axes=(0, 2, 1, 3))
        xk = mx.transpose(xk, axes=(0, 2, 1, 3))
        xv = mx.transpose(xv, axes=(0, 2, 1, 3))

        out = scaled_dot_product_attention_simple(xq, xk, xv, mask=mask)

        # (BATCH_SIZE, H, L, D) -> (BATCH_SIZE, L, H × D)
        out = mx.transpose(out, axes=(0, 2, 1, 3))
        out = mx.reshape(out, (bsz, seqlen, -1))

        return linear(out, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
