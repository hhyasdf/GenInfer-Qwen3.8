"""Numerical verification of the Q8_0 GEMV/GEMM/gather kernels.

Builds random Q8_0 byte tensors, dequantizes them with an independent torch
reference (the closed form from ggml-quants.c dequantize_row_q8_0), and
compares the kernel outputs against torch matmul at bf16 tolerance.

Run: .venv/bin/python tests/test_q80_kernels.py
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")


def dequant_q80_torch(W: torch.Tensor, K: int) -> torch.Tensor:
    """Reference dequant: W [N, K//32*34] uint8 -> [N, K] float32.

    Verified closed form (ggml-quants.c dequantize_row_q8_0):
      per 32-element block: d = half scale (bytes 0..1), qs[32] = int8 values
      y[i*32+j] = qs[j] * d
    """
    N = W.shape[0]
    nb = K // 32
    W = W.view(N, nb, 34)
    d = W[:, :, 0:2].view(torch.float16).to(torch.float32)  # [N, nb, 1]
    qs = W[:, :, 2:34].view(torch.int8).to(torch.float32)  # [N, nb, 32]
    return (qs * d).view(N, K)


def make_q80_bytes(N: int, K: int, seed: int) -> torch.Tensor:
    """Random Q8_0 bytes with plausible value ranges (d ~ N(0,1), qs in [-127,127])."""
    g = torch.Generator().manual_seed(seed)
    nb = K // 32
    qs = torch.randint(-127, 128, (N, nb, 32), generator=g, dtype=torch.int8)
    d = torch.randn(N, nb, generator=g).half()
    out = torch.empty(N, nb, 34, dtype=torch.uint8)
    out[:, :, 0:2] = d.view(torch.uint8).view(N, nb, 2)
    out[:, :, 2:34] = qs.view(torch.uint8)
    return out.view(N, nb * 34)


def check(name: str, got: torch.Tensor, want: torch.Tensor) -> bool:
    got = got.to(torch.float32)
    want = want.to(torch.float32)
    max_abs = (got - want).abs().max().item()
    max_rel = ((got - want).abs() / want.abs().clamp_min(1e-3)).max().item()
    ok = torch.allclose(got, want, rtol=0.008, atol=2.0)
    print(f"  {name}: max_abs={max_abs:.4f} max_rel={max_rel:.4f} -> {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    torch.cuda.set_device(1)  # 2080 Ti (sm_86)
    dev = torch.device("cuda:1")
    from minisgl.kernel import q80

    all_ok = True

    # ---------------- GEMV (decode path: n == 1) ----------------
    print("== GEMV ==", flush=True)
    for K, N in [(5120, 12288), (5120, 1024), (5120, 4096), (17408, 5120), (256, 248320)]:
        Wb = make_q80_bytes(N, K, seed=K + N)
        W = dequant_q80_torch(Wb, K)  # fp32 on CPU
        x = torch.randn(K, generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)
        y = torch.empty(N, device=dev, dtype=torch.bfloat16)
        q80.q80_gemv(x.to(dev), Wb.to(dev), y)
        want = (x.to(torch.float32) @ W.T).to(torch.bfloat16)
        all_ok &= check(f"K={K} N={N}", y.cpu(), want)
        del Wb, W, x, y
        torch.cuda.empty_cache()

    # ---------------- GEMM (prefill path: n > 1) ----------------
    print("== GEMM =", flush=True)
    for M, K, N in [(3, 5120, 1024), (17, 5120, 4096), (128, 5120, 17408), (129, 25600, 5120)]:
        Wb = make_q80_bytes(N, K, seed=M + K + N)
        W = dequant_q80_torch(Wb, K)  # fp32 on CPU
        X = torch.randn(M, K, generator=torch.Generator().manual_seed(11)).to(torch.bfloat16)
        Y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        q80.q80_gemm(X.to(dev), Wb.to(dev), Y)
        want = (X.to(torch.float32) @ W.T).to(torch.bfloat16)
        all_ok &= check(f"M={M} K={K} N={N}", Y.cpu(), want)
        del Wb, W, X, Y
        torch.cuda.empty_cache()

    # ---------------- Gather (embedding path) ----------------
    print("== Gather ==", flush=True)
    for M, K, V in [(5, 256, 248320), (1, 256, 248320), (8, 5120, 1024)]:
        Wb = make_q80_bytes(V, K, seed=M + K + V)
        ids = torch.randint(0, V, (M,), generator=torch.Generator().manual_seed(13))
        W = dequant_q80_torch(Wb[ids], K)  # only the M selected rows
        y = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
        q80.q80_gather(ids.to(dev, torch.int32), Wb.to(dev), y)
        want = W.to(torch.bfloat16)
        all_ok &= check(f"M={M} K={K} V={V}", y.cpu(), want)
        del Wb, W, ids, y
        torch.cuda.empty_cache()

    print("ALL OK" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
