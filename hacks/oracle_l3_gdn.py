"""L3 isolated GDN recurrence oracle (f64 reference vs engine CUDA kernel).

The GDN delta-rule recurrence is the riskiest kernel in the engine (DESIGN.md
section 10). This script draws random q/k/v/g/beta, runs the engine's
``gdn_delta_net`` CUDA kernel, and compares the output and the final state
against a hand-written f64 recurrence that mirrors the kernel line-for-line
(gdn.cu KDA=false path, lines 107-131).

Recurrence (per v-head h, per token t; qk_head = h % Hq):
  kv[col]    = sum_i S[i, col] * k[i]              (S is [i, col], i.e. S^T @ k)
  delta[col] = (v[col] - exp(g) * kv[col]) * beta
  S[i, col]  = exp(g) * S[i, col] + k[i] * delta[col]
  o[col]     = sum_i S_new[i, col] * q[i] * scale  (output uses the NEW state)

The kernel stores the state transposed: state[h, col, i] = S[h, i, col].
"""
import sys

import torch
import torch.nn.functional as F

torch.manual_seed(0)

N_TOKENS = 16
Hq = 16   # num k-heads
Hv = 48   # num v-heads
Sv = 128  # state size (head_k)
SCALE = 1.0 / (Sv ** 0.5)  # 1/sqrt(128)


def ref_recurrence_f64(q, k, v, g, beta, scale):
    """Hand-written f64 recurrence (the primary reference).

    q, k: [1, n, Hq, Sv] f64 (already l2normed, matching the engine's GDNLayer).
    v:    [1, n, Hv, Sv] f64.
    g:    [1, n, Hv] f64 (log-decay, <= 0).
    beta: [1, n, Hv] f64.

    Returns (o [1, n, Hv, Sv] f64, S [Hv, Sv, Sv] f64) where S[h, i, col]
    is the state with i as the first index and col as the second.
    """
    n = q.shape[1]
    S = torch.zeros(Hv, Sv, Sv, dtype=torch.float64)  # S[h, i, col]
    o = torch.zeros(1, n, Hv, Sv, dtype=torch.float64)
    for t in range(n):
        for h in range(Hv):
            qk_head = h % Hq
            q_h = q[0, t, qk_head]  # [Sv]
            k_h = k[0, t, qk_head]  # [Sv]
            v_h = v[0, t, h]  # [Sv]
            g_val = torch.exp(g[0, t, h])
            beta_h = beta[0, t, h]
            # kv[col] = sum_i S[h, i, col] * k[i] = (S[h].T @ k)[col]
            kv = S[h].T @ k_h
            # delta[col] = (v[col] - g * kv[col]) * beta
            delta = (v_h - g_val * kv) * beta_h
            # S_new[i, col] = g * S_old[i, col] + k[i] * delta[col]
            S[h] = g_val * S[h] + torch.outer(k_h, delta)
            # o[col] = sum_i S_new[i, col] * q[i] * scale
            o[0, t, h] = (S[h].T @ q_h) * scale
    return o, S


def main():
    from minisgl.kernel import gdn as gdn_kernel

    device = "cuda:0"
    torch.cuda.set_device(0)

    # 1. Draw random inputs (f64, small values so exp(g) is stable).
    q = torch.randn(1, N_TOKENS, Hq, Sv, dtype=torch.float64) * 0.1
    k = torch.randn(1, N_TOKENS, Hq, Sv, dtype=torch.float64) * 0.1
    v = torch.randn(1, N_TOKENS, Hv, Sv, dtype=torch.float64) * 0.1
    g = -torch.rand(1, N_TOKENS, Hv, dtype=torch.float64) * 0.5  # log-decay in (-0.5, 0)
    beta = torch.rand(1, N_TOKENS, Hv, dtype=torch.float64) * 0.5 + 0.25  # in (0.25, 0.75)

    # 2. L2-norm q/k (the engine's GDNLayer does this before the kernel).
    q_norm = F.normalize(q, dim=-1, eps=1e-6)
    k_norm = F.normalize(k, dim=-1, eps=1e-6)

    # 3. Engine side: cast to f32, call gdn_delta_net.
    q_f = q_norm.float().contiguous().to(device)
    k_f = k_norm.float().contiguous().to(device)
    v_f = v.float().contiguous().to(device)
    g_f = g.float().contiguous().to(device)
    b_f = beta.float().contiguous().to(device)
    dst_f = torch.empty(1, N_TOKENS, Hv, Sv, device=device, dtype=torch.float32)
    S_f = torch.zeros(Hv, Sv, Sv, device=device, dtype=torch.float32)
    gdn_kernel.gdn_delta_net(
        q_f, k_f, v_f, g_f, b_f, dst_f, S_f,
        Hv=Hv, Hq=Hq, n_tokens=N_TOKENS, n_seqs=1, scale=SCALE,
    )
    eng_o = dst_f.cpu().double()
    eng_S = S_f.cpu().double()  # [Hv, col, i] (transposed vs ref S[h, i, col])

    # 4. Reference side (hand-written f64).
    ref_o, ref_S = ref_recurrence_f64(q_norm, k_norm, v, g, beta, SCALE)

    # 5. Compare. The kernel stores the state transposed: state[h, col, i]
    #    = S[h, i, col], so compare eng_S against ref_S.transpose(-1, -2).
    diff_o = (eng_o - ref_o).abs()
    diff_S = (eng_S - ref_S.transpose(-1, -2)).abs()
    print(f"o: max|diff|={diff_o.max():.6f}  mean|diff|={diff_o.mean():.6f}")
    print(f"S: max|diff|={diff_S.max():.6f}  mean|diff|={diff_S.mean():.6f}")
    # f32 accumulation over 128-dim dots: expect ~1e-5..1e-4 abs diff.
    ok = diff_o.max() < 1e-4 and diff_S.max() < 1e-4
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
