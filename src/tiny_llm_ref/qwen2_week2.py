import mlx.core as mx
from .basics import silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights, quantized_linear
from .kv_cache import TinyKvCache


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0, (
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        )
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.rope = RoPE(self.head_dim, max_seq_len, theta)

    def __call__(
        self,
        x: mx.array,
        offsets: list[int],
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        projection_q = quantized_linear(x, self.wq, bias=self.bq).reshape(
            B, L, self.num_heads, self.head_dim
        )
        projection_k = quantized_linear(x, self.wk, bias=self.bk).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        projection_v = quantized_linear(x, self.wv, bias=self.bv).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        # todo: move offsets to kv cache
        if isinstance(offsets, int):
            offset_slice = [slice(int(offsets), int(offsets + L))]
        else:
            offset_slice = [slice(int(i), int(i + L)) for i in offsets]
        projection_q = self.rope(projection_q, offset=offset_slice)
        projection_k = self.rope(projection_k, offset=offset_slice)
        projection_q = projection_q.transpose(0, 2, 1, 3)
        projection_k = projection_k.transpose(0, 2, 1, 3)
        projection_v = projection_v.transpose(0, 2, 1, 3)
        projection_k, projection_v, _, kv_cache_mask = cache.update_and_fetch(
            projection_k, projection_v, q_L=L
        )
        if kv_cache_mask is not None:
            mask = kv_cache_mask
        x = scaled_dot_product_attention_grouped(
            projection_q.astype(mx.float32),
            projection_k.astype(mx.float32),
            projection_v.astype(mx.float32),
            scale=self.scale,
            mask=mask,
        ).astype(x.dtype)
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return quantized_linear(x, self.wo)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        return quantized_linear(
            silu(quantized_linear(x, self.w_gate)) * quantized_linear(x, self.w_up),
            self.w_down,
        )


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        )
        self.self_attn = Qwen2MultiHeadAttention(
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), offset, cache, mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Qwen2ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        precision = mx.float16
        self.precision = precision

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens).astype(precision),
        )
        self.layers_inner = []

        for i in range(mlx_model.args.num_hidden_layers):
            wq = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].self_attn.q_proj
            )
            wk = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].self_attn.k_proj
            )
            wv = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].self_attn.v_proj
            )
            wo = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].self_attn.o_proj
            )
            w_gate = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].mlp.gate_proj
            )
            w_up = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].mlp.up_proj
            )
            w_down = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].mlp.down_proj
            )

            layer = Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=wq,
                wk=wk,
                wv=wv,
                wo=wo,
                bq=mlx_model.model.layers[i].self_attn.q_proj.bias.astype(precision),
                bk=mlx_model.model.layers[i].self_attn.k_proj.bias.astype(precision),
                bv=mlx_model.model.layers[i].self_attn.v_proj.bias.astype(precision),
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                w_input_layernorm=mlx_model.model.layers[
                    i
                ].input_layernorm.weight.astype(precision),
                w_post_attention_layernorm=mlx_model.model.layers[
                    i
                ].post_attention_layernorm.weight.astype(precision),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            self.layers_inner.append(layer)
        self.norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight.astype(precision),
            eps=mlx_model.args.rms_norm_eps,
        )
        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = QuantizedWeights.from_mlx_layer(mlx_model.lm_head)
        else:
            self.w_lm_head = None
        self.mlx_model = mlx_model

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
    ) -> mx.array:
        h = self.embedding(inputs)
        for layer in range(self.num_hidden_layers):
            h = self.layers_inner[layer](h, offset, cache[layer], mask="causal")
        h = self.norm(h)
        if self.w_lm_head is not None:
            return quantized_linear(h, self.w_lm_head)
        else:
            return self.embedding.as_linear(h)
