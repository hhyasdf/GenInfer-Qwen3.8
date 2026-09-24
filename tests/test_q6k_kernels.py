"""Numerical verification of the Q6_K GEMV/GEMM/gather kernels.

Builds random Q6_K byte tensors, dequantizes them with an independent torch
reference (the verified closed form from ggml-quants.c dequantize_row_q6_K),
and compares the kernel outputs against torch matmul at bf16 tolerance.

Run: .venv/bin/python tests/test_q6k_kernels.py
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")


def dequant_q6k_torch(W: torch.Tensor, K: int) -> torch.Tensor:
    """Reference dequant: W [N, K//256*210] uint8 -> [N, K] float32.

    Verified closed form (ggml-quants.c dequantize_row_q6_K):
      per element pos (0..255):
        n = pos // 128; pos_in_sub = pos % 128
        l = pos_in_sub % 32; sub = pos_in_sub // 32; is = l // 16
        ql_idx = n*64 + l + (32 if sub & 1 else 0)
        q_low  = (ql[ql_idx] & 0xF) if sub < 2 else (ql[ql_idx] >> 4)
        q_high = (qh[n*32 + l] >> (sub*2)) & 3
        sc_idx = n*8 + is + sub*2
        w = d * sc[sc_idx] * ((q_low | (q_high << 4)) - 32)
    """
    N = W.shape[0]
    nb = K // 256
    W = W.view(N, nb, 210)
    ql = W[:, :, 0:128].to(torch.int32)
    qh = W[:, :, 128:192].to(torch.int32)
    sc = W[:, :, 192:208].view(torch.int8).to(torch.int32)
    d = W[:, :, 208:210].view(torch.float16).to(torch.float32)  # [N, nb, 1]

    pos = torch.arange(256, dtype=torch.int32)
    n = pos // 128
    pos_in_sub = pos % 128
    l = pos_in_sub % 32
    sub = pos_in_sub // 32
    is_ = l // 16
    ql_idx = n * 64 + l + (sub & 1) * 32
    qh_idx = n * 32 + l
    sc_idx = n * 8 + is_ + sub * 2
    ql_g = torch.gather(ql, 2, ql_idx.expand(N, nb, 256))  # [N, nb, 256]
    qh_g = torch.gather(qh, 2, qh_idx.expand(N, nb, 256))  # [N, nb, 256]
    sc_g = torch.gather(sc, 2, sc_idx.expand(N, nb, 256))  # [N, nb, 256]
    q_low = torch.where(sub[None, None, :] < 2, ql_g & 0xF, ql_g >> 4)
    q_high = (qh_g >> (sub[None, None, :] * 2)) & 3
    q = (q_low | (q_high << 4)) - 32
    w = d * sc_g * q
    return w.view(N, K)


def make_q6k_bytes(N: int, K: int, seed: int) -> torch.Tensor:
    """Random Q6_K bytes with plausible value ranges (d ~ N(0,1), sc in [-64,64])."""
    g = torch.Generator().manual_seed(seed)
    nb = K // 256
    ql = torch.randint(0, 256, (N, nb, 128), generator=g)
    qh = torch.randint(0, 256, (N, nb, 64), generator=g)
    sc = torch.randint(-64, 65, (N, nb, 16), generator=g, dtype=torch.int8)
    d = torch.randn(N, nb, generator=g).half()
    out = torch.empty(N, nb, 210, dtype=torch.uint8)
    out[:, :, 0:128] = ql
    out[:, :, 128:192] = qh
    out[:, :, 192:208] = sc.view(torch.int8)  # 16 int8 scales = 16 bytes
    out[:, :, 208:210] = d.view(torch.uint8).view(N, nb, 2)
    return out.view(N, nb * 210)


def check(name: str, got: torch.Tensor, want: torch.Tensor) -> bool:
    got = got.to(torch.float32)
    want = want.to(torch.float32)
    max_abs = (got - want).abs().max().item()
    max_rel = ((got - want).abs() / want.abs().clamp_min(1e-3)).max().item()
    # Both sides are bf16 roundings of fp32 accumulations. Verified against an
    # fp64 ground truth: the kernel is within ~1 of the truth everywhere and
    # within 1 bf16 ulp of bf16(truth) on 1.7M elements (1 double-straddle).
    # The torch fp32 reference itself deviates from the fp64 truth by up to
    # ~0.15 (MKL accumulation order). Tolerance = 2 bf16 ulp relative
    # (2*2^-8) + 2.0 absolute (reference noise + rounding-boundary straddles).
    # A real kernel bug (wrong dequant index, dropped K-tile) shows up as
    # errors of several ulp on a large fraction of elements — far outside.
    ok = torch.allclose(got, want, rtol=0.008, atol=2.0)
    print(f"  {name}: max_abs={max_abs:.4f} max_rel={max_rel:.4f} -> {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    torch.cuda.set_device(1)  # 2080 Ti (sm_75) — the critical target
    dev = torch.device("cuda:1")
    from minisgl.kernel import q6k

    all_ok = True

    # ---------------- GEMV (decode path: n == 1) ----------------
    print("== GEMV ==", flush=True)
    for K, N in [(5120, 12288), (5120, 10240), (5120, 48), (6144, 5120), (17408, 5120)]:
        Wb = make_q6k_bytes(N, K, seed=K + N)
        W = dequant_q6k_torch(Wb, K)  # fp32 on CPU
        x = torch.randn(K, generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)
        y = torch.empty(N, device=dev, dtype=torch.bfloat16)
        q6k.q6k_gemv(x.to(dev), Wb.to(dev), y)
        want = (x.to(torch.float32) @ W.T).to(torch.bfloat16)
        all_ok &= check(f"K={K} N={N}", y.cpu(), want)
        del Wb, W, x, y
        torch.cuda.empty_cache()

    # ---------------- GEMM (prefill path: n > 1) ----------------
    print("== GEMM =", flush=True)
    for M, K, N in [(3, 5120, 12288), (17, 5120, 10240), (128, 6144, 5120), (129, 17408, 5120)]:
        Wb = make_q6k_bytes(N, K, seed=M + K + N)
        W = dequant_q6k_torch(Wb, K)  # fp32 on CPU
        X = torch.randn(M, K, generator=torch.Generator().manual_seed(11)).to(torch.bfloat16)
        Y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        q6k.q6k_gemm(X.to(dev), Wb.to(dev), Y)
        want = (X.to(torch.float32) @ W.T).to(torch.bfloat16)
        all_ok &= check(f"M={M} K={K} N={N}", Y.cpu(), want)
        del Wb, W, X, Y
        torch.cuda.empty_cache()

    # ---------------- Gather (embedding path) ----------------
    print("== Gather ==", flush=True)
    for M, K, V in [(5, 5120, 248320), (1, 5120, 248320)]:
        Wb = make_q6k_bytes(V, K, seed=M + K + V)  # 1.04 GB CPU
        ids = torch.randint(0, V, (M,), generator=torch.Generator().manual_seed(13))
        W = dequant_q6k_torch(Wb[ids], K)  # only the M selected rows (KB)
        y = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
        q6k.q6k_gather(ids.to(dev, torch.int32), Wb.to(dev), y)
        want = W.to(torch.bfloat16)
        all_ok &= check(f"M={M} K={K} V={V}", y.cpu(), want)
        del Wb, W, ids, y
        torch.cuda.empty_cache()

    print("ALL OK" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
