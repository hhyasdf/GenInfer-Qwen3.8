from __future__ import annotations

import torch
from minisgl.utils import nvtx_annotate

from .base import BaseOP


class Q6KLinear(BaseOP):
    """Linear layer with Q6_K quantized weights kept as raw bytes.

    The weight is stored as a uint8 tensor of shape [N, K // 256 * 210]
    (llama.cpp Q6_K blocks). The CUDA kernels dequantize in registers, so
    every weight byte is read from DRAM exactly once per row.

    forward(x [n, K] bf16) -> [n, N] bf16:
      n == 1  -> Q6_K GEMV (one warp per output row, decode)
      n >  1  -> Q6_K GEMM (128x128 tiles, prefill)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool = False,
    ):
        super().__init__()
        assert input_size % 256 == 0, f"Q6_K requires input_size % 256 == 0, got {input_size}"
        self.input_size = input_size
        self.output_size = output_size
        self.weight = torch.empty(
            output_size, input_size // 256 * 210, dtype=torch.uint8
        )
        self.bias = torch.empty(output_size) if has_bias else None

    @nvtx_annotate("Q6KLinear")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel import q6k

        n, K = x.shape
        assert K == self.input_size, f"expected K={self.input_size}, got {K}"
        N = self.output_size
        if n == 1:
            y = torch.empty(1, N, device=x.device, dtype=x.dtype)
            q6k.q6k_gemv(x.view(K), self.weight, y.view(N))
        else:
            y = torch.empty(n, N, device=x.device, dtype=x.dtype)
            q6k.q6k_gemm(x, self.weight, y)
        if self.bias is not None:
            y = y + self.bias
        return y


class Q6KLMHead(BaseOP):
    """LM head over a Q6_K weight matrix (kept as raw bytes).

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
        assert embedding_dim % 256 == 0, (
            f"Q6_K requires embedding_dim % 256 == 0, got {embedding_dim}"
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.empty(
            num_embeddings, embedding_dim // 256 * 210, dtype=torch.uint8
        )

    @nvtx_annotate("Q6KLMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.core import get_global_ctx
        from minisgl.kernel import q6k

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
            del indices

        n = x.shape[0]
        y = torch.empty(n, self.num_embeddings, device=x.device, dtype=x.dtype)
        if n == 1:
            q6k.q6k_gemv(
                x.view(self.embedding_dim), self.weight, y.view(self.num_embeddings)
            )
        else:
            q6k.q6k_gemm(x, self.weight, y)
        return y


class Q6KEmbedding(BaseOP):
    """Embedding lookup over a Q6_K weight matrix (kept as raw bytes).

    forward(ids [n] int) -> [n, K] bf16 via the Q6_K row-gather kernel.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        assert embedding_dim % 256 == 0, (
            f"Q6_K requires embedding_dim % 256 == 0, got {embedding_dim}"
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.empty(
            num_embeddings, embedding_dim // 256 * 210, dtype=torch.uint8
        )

    @nvtx_annotate("Q6KEmbedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel import q6k

        n = x.shape[0]
        y = torch.empty(n, self.embedding_dim, device=x.device, dtype=torch.bfloat16)
        q6k.q6k_gather(x.to(torch.int32), self.weight, y)
        return y
