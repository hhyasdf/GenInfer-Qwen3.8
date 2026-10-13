"""Q6_K kernels (ported from llama.cpp ggml-cuda).

Q6_K weights are stored as uint8 byte tensors (210 bytes per 256-element
block). The GEMV kernel (decode) and GEMM kernel (prefill) dequantize in
registers; the gather kernel dequantizes whole rows for embedding lookup.
All activations are bf16.
"""

import functools

import torch

from .utils import KernelConfig, load_jit, make_cpp_args


def _bytes_per_block(K: int) -> int:
    assert K % 256 == 0, f"Q6_K requires K to be a multiple of 256, got {K}"
    return K // 256 * 210


@functools.cache
def _gemv_module(K: int, N: int, f32_out: bool):
    config = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)
    args = make_cpp_args(K, N, *config, f32_out)
    return load_jit(
        "q6k_gemv",
        *args,
        cuda_files=["q6k.cu"],
        cuda_wrappers=[("run", f"Q6KGemvKernel<{args}>::run")],
    )


@functools.cache
def _gemv_batch_module(K: int, N: int, M: int, f32_out: bool):
    config = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)
    args = make_cpp_args(K, N, M, *config, f32_out)
    return load_jit(
        "q6k_gemv_batch",
        *args,
        cuda_files=["q6k.cu"],
        cuda_wrappers=[("run", f"Q6KGemvBatchKernel<{args}>::run")],
    )


@functools.cache
def _gemm_module(K: int, N: int, f32_out: bool):
    config = KernelConfig(num_threads=256, max_occupancy=1, use_pdl=False)
    args = make_cpp_args(K, N, *config, f32_out)
    return load_jit(
        "q6k_gemm",
        *args,
        cuda_files=["q6k.cu"],
        cuda_wrappers=[("run", f"Q6KGemmKernel<{args}>::run")],
    )


@functools.cache
def _gather_module(K: int):
    config = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)
    args = make_cpp_args(K, *config)
    return load_jit(
        "q6k_gather",
        *args,
        cuda_files=["q6k.cu"],
        cuda_wrappers=[("run", f"Q6KGatherKernel<{args}>::run")],
    )


def q6k_gemv(
    x: torch.Tensor,  # [K] bf16
    W: torch.Tensor,  # [N, K // 256 * 210] uint8 (Q6_K)
    y: torch.Tensor,  # [N] bf16 or f32
) -> None:
    """Q6_K GEMV: y[n] = sum_k W[n, k] * x[k]. One warp per output row.

    Output dtype is taken from ``y``: bf16 for activations, f32 for logits.
    """
    K = x.shape[0]
    N = y.shape[0]
    _gemv_module(K, N, y.dtype == torch.float32).run(x, W, y)


def q6k_gemv_batch(
    X: torch.Tensor,  # [M, K] bf16
    W: torch.Tensor,  # [N, K // 256 * 210] uint8 (Q6_K)
    Y: torch.Tensor,  # [M, N] bf16 or f32
) -> None:
    """Q6_K batched GEMV: Y[m, n] = sum_k X[m, k] * W[n, k].

    One warp per output row accumulates all M input rows while reading each
    weight row exactly once. For the small M produced by spec-decode verify
    (M = K_spec+1, e.g. 5) this reads the weights ~M times less than M
    separate q6k_gemv launches, which dominate the small-batch verify step.
    Output dtype is taken from ``Y``: bf16 for activations, f32 for logits.
    """
    M, K = X.shape
    N = Y.shape[1]
    _gemv_batch_module(K, N, M, Y.dtype == torch.float32).run(X, W, Y)


def q6k_gemm(
    X: torch.Tensor,  # [M, K] bf16
    W: torch.Tensor,  # [N, K // 256 * 210] uint8 (Q6_K)
    Y: torch.Tensor,  # [M, N] bf16 or f32
) -> None:
    """Q6_K GEMM: Y[m, n] = sum_k X[m, k] * W[n, k]. 128x128 tiles, N-fast grid.

    Output dtype is taken from ``Y``: bf16 for activations, f32 for logits.
    """
    M, K = X.shape
    N = Y.shape[1]
    _gemm_module(K, N, Y.dtype == torch.float32).run(X, W, Y)


def q6k_gather(
    ids: torch.Tensor,  # [M] int32
    W: torch.Tensor,  # [V, K // 256 * 210] uint8 (Q6_K)
    y: torch.Tensor,  # [M, K] bf16
) -> None:
    """Q6_K row gather: y[m, :] = dequant(W[ids[m], :]). For embedding lookup."""
    M = ids.shape[0]
    K = y.shape[1]
    _gather_module(K).run(ids, W, y)
