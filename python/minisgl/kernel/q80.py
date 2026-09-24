"""Q8_0 kernels (ported from llama.cpp ggml-cuda).

Q8_0 weights are stored as uint8 byte tensors (34 bytes per 32-element
block: 2-byte half scale + 32 int8 values). The GEMV kernel (decode) and
GEMM kernel (prefill) dequantize in registers; the gather kernel
dequantizes whole rows for embedding lookup. All activations are bf16.
"""

import functools

import torch

from .utils import KernelConfig, load_jit, make_cpp_args


def _bytes_per_block(K: int) -> int:
    assert K % 32 == 0, f"Q8_0 requires K to be a multiple of 32, got {K}"
    return K // 32 * 34


@functools.cache
def _gemv_module(K: int, N: int):
    config = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)
    args = make_cpp_args(K, N, *config)
    return load_jit(
        "q80_gemv",
        *args,
        cuda_files=["q80.cu"],
        cuda_wrappers=[("run", f"Q80GemvKernel<{args}>::run")],
    )


@functools.cache
def _gemm_module(K: int, N: int):
    config = KernelConfig(num_threads=256, max_occupancy=1, use_pdl=False)
    args = make_cpp_args(K, N, *config)
    return load_jit(
        "q80_gemm",
        *args,
        cuda_files=["q80.cu"],
        cuda_wrappers=[("run", f"Q80GemmKernel<{args}>::run")],
    )


@functools.cache
def _gather_module(K: int):
    config = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)
    args = make_cpp_args(K, *config)
    return load_jit(
        "q80_gather",
        *args,
        cuda_files=["q80.cu"],
        cuda_wrappers=[("run", f"Q80GatherKernel<{args}>::run")],
    )


def q80_gemv(
    x: torch.Tensor,  # [K] bf16
    W: torch.Tensor,  # [N, K // 32 * 34] uint8 (Q8_0)
    y: torch.Tensor,  # [N] bf16
) -> None:
    """Q8_0 GEMV: y[n] = sum_k W[n, k] * x[k]. One warp per output row."""
    K = x.shape[0]
    N = y.shape[0]
    _gemv_module(K, N).run(x, W, y)


def q80_gemm(
    X: torch.Tensor,  # [M, K] bf16
    W: torch.Tensor,  # [N, K // 32 * 34] uint8 (Q8_0)
    Y: torch.Tensor,  # [M, N] bf16
) -> None:
    """Q8_0 GEMM: Y[m, n] = sum_k X[m, k] * W[n, k]. 128x128 tiles, N-fast grid."""
    M, K = X.shape
    N = Y.shape[1]
    _gemm_module(K, N).run(X, W, Y)


def q80_gather(
    ids: torch.Tensor,  # [M] int32
    W: torch.Tensor,  # [V, K // 32 * 34] uint8 (Q8_0)
    y: torch.Tensor,  # [M, K] bf16
) -> None:
    """Q8_0 row gather: y[m, :] = dequant(W[ids[m], :]). For embedding lookup."""
    M = ids.shape[0]
    K = y.shape[1]
    _gather_module(K).run(ids, W, y)
