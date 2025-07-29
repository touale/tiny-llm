import mlx.core as mx
from tiny_llm.basics import linear, silu
from tiny_llm.attention import scaled_dot_product_attention_grouped
from tiny_llm.layer_norm import RMSNorm
from tiny_llm.positional_encoding import RoPE
from typing import Any
from tiny_llm.embedding import Embedding
from tiny_llm.quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.head_dim = hidden_size // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = linear(x, self.wq, bias=self.bq)
        k = linear(x, self.wk, bias=self.bk)
        v = linear(x, self.wv, bias=self.bv)

        xq = q.reshape(B, L, self.num_heads, self.head_dim)
        xk = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        xv = v.reshape(B, L, self.num_kv_heads, self.head_dim)

        rope = RoPE(self.head_dim, self.max_seq_len, base=self.theta, traditional=False)
        xq = rope(xq, offset=slice(offset, offset + x.shape[1]))
        xk = rope(xk, offset=slice(offset, offset + x.shape[1]))

        xq = xq.transpose(0, 2, 1, 3)
        xk = xk.transpose(0, 2, 1, 3)
        xv = xv.transpose(0, 2, 1, 3)

        x = scaled_dot_product_attention_grouped(xq, xk, xv, None, mask)

        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(B, L, -1)
        x = linear(x, self.wo)

        return x


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        pass
