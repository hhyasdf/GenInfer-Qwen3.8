"""Compute f64 reference layer-3 full-attn intermediates (qg, q, k, v, o).

Input x = rmsnorm(engine layer_02_r, input_layernorm_3) — the residual after
layer 2 (GDN layers 0-2 match, so the engine's layer_02_r is the correct input).
"""
from __future__ import annotations
import json, time
import numpy as np
import torch

OUT = "/tmp/bisect"
W = f"{OUT}/w"

def load_w(name):
    parts = []
    c = 0
    while True:
        p = f"{W}/{name}_c{c}.pt"
        import os
        if not os.path.exists(p):
            break
        parts.append(torch.load(p, map_location="cpu"))
        c += 1
    assert parts, f"missing weight file {name}"
    return torch.cat(parts, dim=0).float().numpy().astype(np.float64)

def rmsnorm(x, w, eps):
    rms = np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return x / rms * w

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
    t0 = time.time()
    meta = json.loads(open(f"{OUT}/meta.json").read())
    eps = meta["eps"]
    small = torch.load(f"{OUT}/small.pt", map_location="cpu")
    caps = torch.load(f"{OUT}/captures.pt", map_location="cpu")

    # input to layer 3 = rmsnorm(layer_02_r + h_2, input_layernorm_3)
    # (the decoder layer adds the previous MLP output to the residual before norm)
    r2 = caps["layer_02_r"].float().numpy().astype(np.float64)
    h2 = caps["layer_02_h"].float().numpy().astype(np.float64)
    iln3 = small["L03"]["input_layernorm"].float().numpy().astype(np.float64)
    x = rmsnorm(r2 + h2, iln3, eps)
    n = x.shape[0]
    print(f"x: {x.shape}, max|.|={np.abs(x).max():.4f}", flush=True)

    W3 = {
        "attn_q": load_w("L03_attn_q"),
        "attn_k": load_w("L03_attn_k"),
        "attn_v": load_w("L03_attn_v"),
        "attn_output": load_w("L03_attn_output"),
        "attn_q_norm": small["L03"]["attn_q_norm"].float().numpy().astype(np.float64),
        "attn_k_norm": small["L03"]["attn_k_norm"].float().numpy().astype(np.float64),
    }

    # QG projection + per-head interleaved split
    qg = x @ W3["attn_q"].T  # [n, 12288]
    qg_r = qg.reshape(n, 24, 512)
    q = qg_r[:, :, :256].copy()
    gate = qg_r[:, :, 256:].copy()
    # per-head q norm
    q = rmsnorm(q, W3["attn_q_norm"], eps)
    # K projection + per-head k norm
    k = (x @ W3["attn_k"].T).reshape(n, 4, 256)
    k = rmsnorm(k, W3["attn_k_norm"], eps)
    # V projection
    v = (x @ W3["attn_v"].T).reshape(n, 4, 256)

    # partial RoPE (rope_dim=64, base=1e7)
    rope_dim = 64
    rope_base = 1e7
    half = rope_dim // 2
    positions = np.arange(n, dtype=np.float64)
    theta = rope_base ** (-2.0 * np.arange(half) / rope_dim)
    ang = positions[:, None] * theta[None, :]
    cos = np.cos(ang)[:, None, :]
    sin = np.sin(ang)[:, None, :]

    def rope(xh):
        x1 = xh[..., :half].copy()
        x2 = xh[..., half:2*half].copy()
        out = xh.copy()
        out[..., :half] = x1 * cos - x2 * sin
        out[..., half:2*half] = x2 * cos + x1 * sin
        return out

    q_rope = rope(q)
    k_rope = rope(k)

    # causal GQA attention, scale 1/16, kv head = q head // 6
    k_e = np.repeat(k_rope, 6, axis=1)
    v_e = np.repeat(v, 6, axis=1)
    scores = np.einsum("thd,shd->ths", q_rope, k_e) / 16.0
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(mask[:, None, :], -1e30, scores)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    o = np.einsum("ths,shd->thd", p, v_e)  # [n, 24, 256]
    o = o * sigmoid(gate)

    print(f"qg: {qg.shape} max|.|={np.abs(qg).max():.4f}", flush=True)
    print(f"q (rope): {q_rope.shape} max|.|={np.abs(q_rope).max():.4f}", flush=True)
    print(f"k (rope): {k_rope.shape} max|.|={np.abs(k_rope).max():.4f}", flush=True)
    print(f"v: {v.shape} max|.|={np.abs(v).max():.4f}", flush=True)
    print(f"gate: {gate.shape} max|.|={np.abs(gate).max():.4f}", flush=True)
    print(f"o (after gate): {o.shape} max|.|={np.abs(o).max():.4f}", flush=True)

    np.savez(
        f"{OUT}/ref_l3_inter.npz",
        x=x, qg=qg, q=q_rope, k=k_rope, v=v, gate=gate, o=o,
    )
    print(f"saved ref_l3_inter.npz ({time.time()-t0:.1f} s)", flush=True)

if __name__ == "__main__":
    main()
