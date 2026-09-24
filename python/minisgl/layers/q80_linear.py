from __future__ import annotations

import torch
from minisgl.utils import nvtx_annotate

from .base import BaseOP


class Q80Linear(BaseOP):
    """Linear layer with Q8_0 quantized weights kept as raw bytes.

    The weight is stored as a uint8 tensor of shape [N, K // 32 * 34]
    (llama.cpp Q8_0 blocks: 2-byte half scale + 32 int8 values). The CUDA
    kernels dequantize in registers, so every weight byte is read from DRAM
    exactly once per row.

    forward(x [n, K] bf16) -> [n, N] bf16:
      n == 1  -> Q8_0 GEMV (one warp per output row, decode)
      n >  1  -> Q8_0 GEMM (128x128 tiles, prefill)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool = False,
    ):
        super().__init__()
        assert input_size % 32 == 0, f"Q8_0 requires input_size % 32 == 0, got {input_size}"
        self.input_size = input_size
        self.output_size = output_size
        self.weight = torch.empty(
            output_size, input_size // 32 * 34, dtype=torch.uint8
        )
        self.bias = torch.empty(output_size) if has_bias else None

    @nvtx_annotate("Q80Linear")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel import q80

        n, K = x.shape
        assert K == self.input_size, f"expected K={self.input_size}, got {K}"
        N = self.output_size
        if n == 1:
            y = torch.empty(1, N, device=x.device, dtype=x.dtype)
            q80.q80_gemv(x.view(K), self.weight, y.view(N))
        else:
            y = torch.empty(n, N, device=x.device, dtype=x.dtype)
            q80.q80_gemm(x, self.weight, y)
        if self.bias is not None:
            y = y + self.bias
        return y


class Q80LMHead(BaseOP):
    """LM head over a Q8_0 weight matrix (kept as raw bytes).

    In prefill only the last token of each sequence is projected (the
    scaffold convention); n == 1 dispatches to the GEMV kernel, n > 1 to
    the GEMM kernel.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        assert embedding_dim % 32 == 0, (
            f"Q8_0 requires embedding_dim % 32 == 0, got {embedding_dim}"
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.empty(
            num_embeddings, embedding_dim // 32 * 34, dtype=torch.uint8
        )

    @nvtx_annotate("Q80LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.core import get_global_ctx
        from minisgl.kernel import q80

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
            del indices

        n = x.shape[0]
        y = torch.empty(n, self.num_embeddings, device=x.device, dtype=x.dtype)
        if n == 1:
            q80.q80_gemv(
                x.view(self.embedding_dim), self.weight, y.view(self.num_embeddings)
            )
        else:
            q80.q80_gemm(x, self.weight, y)
        return y


class Q80Embedding(BaseOP):
    """Embedding lookup over a Q8_0 weight matrix (kept as raw bytes).

    forward(ids [n] int) -> [n, K] bf16 via the Q8_0 row-gather kernel.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        assert embedding_dim % 32 == 0, (
            f"Q8_0 requires embedding_dim % 32 == 0, got {embedding_dim}"
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.empty(
            num_embeddings, embedding_dim // 32 * 34, dtype=torch.uint8
        )

    @nvtx_annotate("Q80Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel import q80

        n = x.shape[0]
        y = torch.empty(n, self.embedding_dim, device=x.device, dtype=torch.bfloat16)
        q80.q80_gather(x.to(torch.int32), self.weight, y)
        return y
