"""Isolate the layer-3 divergence: output projection vs MLP.

Both the f64 ref and the engine use the SAME dequantized weights (w/*.pt),
so a divergence here is a *computation* difference, not a weight bug.

We compute the f64 ref's post-`o` steps from the dumped `o` and compare:
  r_ref = (layer_02_r + layer_02_h) + o @ attn_output.T   vs  engine layer_03_r
  h_ref = MLP(rmsnorm(r_ref))                              vs  engine layer_03_h
If r_ref diverges  -> output projection (attn_output GEMM) is the bug.
If r_ref matches but h_ref diverges -> the MLP (or post_attention_layernorm) is the bug.
"""
import json
import numpy as np
import torch

OUT = "/tmp/bisect"
W = f"{OUT}/w"
meta = json.loads(open(f"{OUT}/meta.json").read())
eps = meta["eps"]

def load_w(name):
    parts, c = [], 0
    while True:
        import os
        p = f"{W}/{name}_c{c}.pt"
        if not os.path.exists(p):
            break
        parts.append(torch.load(p, map_location="cpu"))
        c += 1
    return torch.cat(parts, dim=0).float().numpy().astype(np.float64)

def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

def silu(x):
    return x / (1.0 + np.exp(-x))

def cmp(name, ref, eng, tol_rel=0.02):
    ref = np.asarray(ref, dtype=np.float64)
    eng = np.asarray(eng, dtype=np.float64)
    if ref.shape != eng.shape:
        print(f"SHAPE   {name:12s} ref={ref.shape} eng={eng.shape}")
        return
    d = np.abs(ref - eng)
    mx = float(d.max()); mean = float(d.mean())
    scale = max(float(np.abs(ref).max()), 1e-9)
    rel = mx / scale
    frac = float((d > 0.1 * np.abs(ref)).mean())
    verdict = "OK" if rel <= tol_rel and frac < 0.05 else "DIVERGE"
    print(f"{verdict:7s} {name:12s} max|diff|={mx:.5f}  mean={mean:.5f}  rel={rel:.4e}  frac={frac:.4f}")

# ---- load ----
ref = np.load(f"{OUT}/ref_l3_inter.npz")
o_ref = ref["o"].astype(np.float64)          # [73, 24, 256] gated attn out (f64)
caps = torch.load(f"{OUT}/captures.pt", map_location="cpu")
small = torch.load(f"{OUT}/small.pt", map_location="cpu")

r2 = caps["layer_02_r"].float().numpy().astype(np.float64)   # [73, 5120]
h2 = caps["layer_02_h"].float().numpy().astype(np.float64)   # [73, 5120]
eng_r3 = caps["layer_03_r"].float().numpy().astype(np.float64)
eng_h3 = caps["layer_03_h"].float().numpy().astype(np.float64)

attn_output = load_w("L03_attn_output")      # [5120, 6144]
mlp_gate = load_w("L03_mlp_gate")            # [17408, 5120]
mlp_up = load_w("L03_mlp_up")                # [17408, 5120]
mlp_down = load_w("L03_mlp_down")            # [5120, 17408]
paln = small["L03"]["post_attention_layernorm"].float().numpy().astype(np.float64)

print(f"o_ref {o_ref.shape} max|.|={np.abs(o_ref).max():.4f}")
print(f"attn_output {attn_output.shape}  mlp_gate {mlp_gate.shape}  mlp_down {mlp_down.shape}")

# ---- f64 post-`o` steps ----
a_ref = o_ref.reshape(o_ref.shape[0], -1) @ attn_output.T     # [73, 5120]
r_ref = (r2 + h2) + a_ref                                     # [73, 5120]
x2 = rmsnorm(r_ref, paln, eps)                                # [73, 5120]
g = x2 @ mlp_gate.T                                           # [73, 17408]
u = x2 @ mlp_up.T                                             # [73, 17408]
h_ref = (silu(g) * u) @ mlp_down.T                            # [73, 5120]

print(f"a_ref (out proj) {a_ref.shape} max|.|={np.abs(a_ref).max():.4f}")
print(f"r_ref {r_ref.shape} max|.|={np.abs(r_ref).max():.4f}")
print(f"h_ref {h_ref.shape} max|.|={np.abs(h_ref).max():.4f}")

print("\n=== f64-ref post-`o` vs engine captures ===")
cmp("layer_03_r", r_ref, eng_r3)
cmp("layer_03_h", h_ref, eng_h3)

# Also: what does the engine's OWN `o` (bf16->f32) give for the out proj?
eng = torch.load(f"{OUT}/eng_l3_inter.pt", map_location="cpu")
o_eng = eng["o"].float().numpy().astype(np.float64)
a_eng_o = o_eng.reshape(o_eng.shape[0], -1) @ attn_output.T
r_eng_o = (r2 + h2) + a_eng_o
print("\n=== using engine's own `o` (bf16) for f64 out-proj ===")
cmp("layer_03_r(o_eng)", r_eng_o, eng_r3)
