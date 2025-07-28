import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

        # build the cache
        seq_idx = mx.arange(0, seq_len, dtype=mx.float32)

        # Get θ
        theta = 1 / (
            self.base
            ** (
                mx.arange(0, self.dims, 2, dtype=mx.float32)[: self.dims // 2]
                / self.dims
            )
        )

        # Got mθ
        idx_theta = mx.einsum("i, j -> ij", seq_idx, theta)

        # [cos, sin] cache
        self.cache = mx.stack([mx.cos(idx_theta), mx.sin(idx_theta)], axis=-1)
        print(f"RoPE cache shape: {self.cache.shape}")

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        """
        x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
        """
        seq_len = x.shape[1]

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.reshape((*x.shape[:-1], -1, 2))

        cache = self.cache[offset] if offset else self.cache[:seq_len]
        rope_cache = cache.reshape((-1, xshaped.shape[1], 1, xshaped.shape[3], 2))

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        if self.traditional:
            x1 = xshaped[..., ::2].reshape(*xshaped.shape[:-1])
            x2 = xshaped[..., 1::2].reshape(*xshaped.shape[:-1])
        else:
            x1, x2 = mx.split(x, 2, axis=-1)
            x1 = x1.reshape(*xshaped.shape[:-1])
            x2 = x2.reshape(*xshaped.shape[:-1])
            print(
                f"RoPE x1 shape: {x1.shape}, x2 shape: {x2.shape}, xshaped shape: {xshaped.shape}"
            )

        o1 = x1 * rope_cache[..., 0] - x2 * rope_cache[..., 1]
        o2 = x2 * rope_cache[..., 0] + x1 * rope_cache[..., 1]

        if self.traditional:
            out = mx.stack(
                [o1, o2],
                axis=-1,
            ).flatten(start_axis=-2)
        else:
            out = mx.concatenate((o1, o2), axis=-1)
        return out.astype(x.dtype)
