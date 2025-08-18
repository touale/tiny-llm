import mlx.core as mx


def make_sampler(temp: float, top_p: float | None=None, top_k: int | None = None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        else:
            if top_k:
                mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
                logprobs = mx.put_along_axis(
                    logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
                )
            if top_p:
                sorted_indices = mx.argsort(logprobs, axis=-1)
                sorted_logprobs = logprobs[:, sorted_indices]
                cumsum = mx.cumsum(mx.exp(sorted_logprobs), axis=-1)
                mask_elements = cumsum < top_p
                mask_elements[..., 0] = True
                logprobs[:, sorted_indices] = mx.where(mask_elements, sorted_logprobs, -mx.inf)

            logprobs = logprobs / temp
            return mx.random.categorical(logprobs,axis=-1)
        

    return sample
