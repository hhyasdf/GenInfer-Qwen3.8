"""Compare f64 ref vs engine layer-3 full-attn intermediates."""
import numpy as np
import torch

ref = np.load("/tmp/bisect/ref_l3_inter.npz")
eng = torch.load("/tmp/bisect/eng_l3_inter.pt", map_location="cpu")

def cmp(name, r, e):
    r = np.asarray(r, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    if r.shape != e.shape:
        print(f"SHAPE   {name:10s} ref={r.shape} eng={e.shape}")
        return
    d = np.abs(r - e)
    mx = float(d.max())
    mean = float(d.mean())
    scale = max(float(np.abs(r).max()), 1e-9)
    rel = mx / scale
    frac = float((d > 0.1 * np.abs(r)).mean())
    verdict = "OK" if rel <= 0.02 and frac < 0.05 else "DIVERGE"
    print(f"{verdict:7s} {name:10s} max|diff|={mx:.5f}  mean={mean:.5f}  rel={rel:.4e}  frac(|d|>0.1|ref|)={frac:.4f}")

print("=== layer-3 full-attn intermediates: f64 ref vs engine ===")
cmp("x", ref["x"], eng["x"])
cmp("qg", ref["qg"], eng["qg"])
cmp("q", ref["q"], eng["q"])
cmp("k", ref["k"], eng["k"])
cmp("v", ref["v"], eng["v"])
cmp("gate", ref["gate"], eng["gate"])
cmp("o", ref["o"], eng["o"])
print("positions match:", np.array_equal(ref["x"].shape[0], eng["positions"].shape[0]))
print("eng positions:", eng["positions"][:5].tolist(), "...", eng["positions"][-3:].tolist())
