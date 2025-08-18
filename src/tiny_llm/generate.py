import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from tiny_llm.kv_cache import TinyKvFullCache
from tiny_llm.qwen2_week1 import Qwen2ModelWeek1
from tiny_llm.qwen2_week2 import Qwen2ModelWeek2
from typing import Callable

from tiny_llm.sampler import make_sampler


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model: Qwen2ModelWeek1, tokens: list[int], offset):
        logits = model(tokens, offset)[:, -1, :]
        token_id = sampler(logits)

        # new_input_ids = mx.concatenate([tokens, token_id], axis=-1)
        return token_id[:, None]

    tokens = mx.array([tokenizer.encode(prompt)], dtype=mx.uint32)
    new_token = None
    offset = 0
    while new_token != tokenizer.eos_token_id:
        new_token = _step(model, tokens, offset)
        tokens = mx.concatenate([tokens, new_token], axis=-1)
        offset = tokens.shape[-1] + 1

        print(tokenizer.decode(tokens.tolist()[0]))
    return tokenizer.decode(tokens.tolist()[0])



def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]

    def _step(model, y, offset, kv_cache):
        logits = model(y[None], offset, kv_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        sampler = lambda x: mx.argmax(x, axis=-1)
        y = sampler(logprobs)
        return y, logprobs.squeeze(0)

    # prefill with the prompt
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    offset = tokens.size
    # generate/decode
    while True:
        print(f"Current offset: {offset}, tokens: {tokens}")
        token, _ = _step(model, tokens, offset, kv_cache)
        detokenizer.add_token(token.item())
        if token.item() == tokenizer.eos_token_id:
            break
        offset += tokens.size
        tokens = token
        



def batch_generate(
    model: any,
    tokenizer: TokenizerWrapper,
    prompts: list[str],
    max_seq_len=512,
    batch_size=5,
    prefill_step=128,
):
    pass

if __name__ == "__main__":
    from mlx_lm import load

    mlx_model, tokenizer = load( "Qwen/Qwen2-0.5B-Instruct-MLX")
    model = Qwen2ModelWeek2(mlx_model)
    simple_generate_with_kv_cache(model, tokenizer , "你好呀，你是谁？")
