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

        x = scaled_dot_product_attention_grouped(
            xq.astype(mx.float32),
            xk.astype(mx.float32),
            xv.astype(mx.float32),
            None,
            mask,
        ).astype(x.dtype)

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
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        w_gate = linear(x, self.w_gate)
        w_up = linear(x, self.w_up)
        out = silu(w_gate)
        out = w_up * out
        return linear(out, self.w_down)


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
        self.attention = Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
        )
        self.mlp = Qwen2MLP(
            dim=hidden_size,
            hidden_dim=hidden_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
        )
        self.input_norm = RMSNorm(
            dim=hidden_size, weight=w_input_layernorm, eps=rms_norm_eps
        )
        self.post_norm = RMSNorm(
            dim=hidden_size, weight=w_post_attention_layernorm, eps=rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        o1 = self.input_norm(x)
        o1 = self.attention(o1, offset=offset, mask=mask)
        o1 = x + o1
        o2 = self.post_norm(o1)
        o2 = self.mlp(o2)
        return o2 + o1


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        dtype = mx.float16
        self.args = mlx_model.args
        self.layers = [
            Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=dequantize_linear(layer.self_attn.q_proj).astype(dtype),
                wk=dequantize_linear(layer.self_attn.k_proj).astype(dtype),
                wv=dequantize_linear(layer.self_attn.v_proj).astype(dtype),
                wo=dequantize_linear(layer.self_attn.o_proj).astype(dtype),
                bq=layer.self_attn.q_proj.bias.astype(dtype),
                bk=layer.self_attn.k_proj.bias.astype(dtype),
                bv=layer.self_attn.v_proj.bias.astype(dtype),
                w_gate=dequantize_linear(layer.mlp.gate_proj).astype(dtype),
                w_up=dequantize_linear(layer.mlp.up_proj).astype(dtype),
                w_down=dequantize_linear(layer.mlp.down_proj).astype(dtype),
                w_input_layernorm=layer.input_layernorm.weight.astype(dtype),
                w_post_attention_layernorm=layer.post_attention_layernorm.weight.astype(
                    dtype
                ),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            for i in range(mlx_model.args.num_hidden_layers)
            if (layer := mlx_model.layers[i])
        ]
        self.emb = Embedding(
            vocab_size=mlx_model.args.vocab_size,
            embedding_dim=mlx_model.args.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens).astype(mx.float16),
        )
        self.norm = RMSNorm(
            dim=mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight.astype(mx.float16),
            eps=mlx_model.args.rms_norm_eps,
        )
        if mlx_model.args.tie_word_embeddings:
            self.out_layer = lambda x: self.emb.as_linear(x)
        else:
            self.out_layer = lambda x: linear(x, dequantize_linear(mlx_model.lm_head))

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        out = self.emb(inputs)
        for layer in self.layers:
            out = layer(out, offset=offset, mask="causal" if out.shape[1] > 1 else None)
        out = self.norm(out)
        out = self.out_layer(out)
        return out
