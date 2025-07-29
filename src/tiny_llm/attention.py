import mlx.core as mx
from tiny_llm.basics import softmax, linear
import math
import numpy as np


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
    mask = mx.triu(mx.ones((L, S)), k=S - L + 1)
    mask = mx.where(mask, mx.array(-mx.inf), mx.array(0)).astype(dtype)
    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    """
    query: N.. x H_q x L x D
    key: N.. x H x S x D
    value: N.. x H x S x D
    mask: N.. x H_q x L x S
    output: N.. x H_q x L x D
    """
    origin_shape = query.shape
    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]
    assert H_q % H == 0
    n_repeats = H_q // H

    query = query.reshape(-1, H, n_repeats, L, D)
    key = key.reshape(-1, H, 1, S, D)
    value = value.reshape(-1, H, 1, S, D)

    if mask is not None:
        if mask == "causal":
            mask = causal_mask(L, S, query.dtype)
        else:
            mask = mask.reshape(-1, H, n_repeats, mask.shape[-2], mask.shape[-1])

    out = scaled_dot_product_attention_simple(query, key, value, scale, mask)
    return out.reshape(origin_shape)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass


H_q = 18
H = 6
L = 7
D = 5
S = 7
BATCH = 10
BATCH_2 = 2
precision = mx.float32

q_shape = (H_q, L, D)
kv_shape = (H, S, D)
mask_shape = (H_q, L, S)
query = mx.random.uniform(shape=q_shape, dtype=precision)
key = mx.random.uniform(shape=kv_shape, dtype=precision)
value = mx.random.uniform(shape=kv_shape, dtype=precision)
mask = mx.random.uniform(shape=mask_shape, dtype=precision)

reference_output = scaled_dot_product_attention_grouped(
    query.reshape(-1, H_q, L, D),
    key.reshape(-1, H, S, D),
    value.reshape(-1, H, S, D),
    # scale=scale if scale is not None else (1.0 / (D**0.5)),
    mask="causal",
)
