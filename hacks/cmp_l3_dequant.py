"""Cross-check L03_attn_output dequant: independent numpy f64 vs engine q6k_gather (w/*.pt).

If they match  -> the dequant is correct; the layer-3 divergence is in q6k_gemm (the GEMM).
If they differ -> the dequant of attn_output is the bug.
Also recompute the f64 output projection with the INDEPENDENT dequant and compare
to the engine's layer_03_r to see which dequant the engine's GEMM actually matches.
"""
import json
import sys
import numpy as np
import torch
from pathlib import Path

# The engine root is the parent of hacks/; add it so the sibling hack
# (bisect_cpu) and the engine package import from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bisect_cpu import parse_gguf, dequant_rows, read_rows, rmsnorm

ROOT = Path("/home/hhy/Projects/GenInfer-Qwen3.8")
OUT = Path("/tmp/bisect")
W = OUT / "w"
CHUNK = 16384

import tomllib
from minisgl.models.gguf_weight import find_main_gguf
cfg = tomllib.load(open(ROOT / "config.toml", "rb"))
gguf_path = find_main_gguf(cfg["engine"]["model"])
print(f"GGUF: {gguf_path}")

meta = json.loads((OUT / "meta.json").read_text())
eps = meta["eps"]

gmeta, gtensors, data_start, mm = parse_gguf(Path(gguf_path))
gmap = {t["name"]: t for t in gtensors}

def cmp(name, ref, eng, tol_rel=0.02):
    ref = np.asarray(ref, dtype=np.float64)
    eng = np.asarray(eng, dtype=np.float64)
    if ref.shape != eng.shape:
        print(f"SHAPE   {name:16s} ref={ref.shape} eng={eng.shape}")
        return
    d = np.abs(ref - eng)
    mx = float(d.max()); mean = float(d.mean())
    scale = max(float(np.abs(ref).max()), 1e-9)
    rel = mx / scale
    frac = float((d > 0.1 * np.abs(ref)).mean())
    verdict = "OK" if rel <= tol_rel and frac < 0.05 else "DIVERGE"
    print(f"{verdict:7s} {name:16s} max|diff|={mx:.5f}  mean={mean:.5f}  rel={rel:.4e}  frac={frac:.4f}")

# ---- independent dequant of blk.3.attn_output.weight ----
tinfo = gmap["blk.3.attn_output.weight"]
n_rows = tinfo["dims"][0]
row_bytes = tinfo["dims"][1] // 256 * 210
print(f"blk.3.attn_output.weight dims={tinfo['dims']} type={tinfo['type']} row_bytes={row_bytes} n_rows={n_rows}")

# full independent dequant (5120 rows x 6144) in chunks to bound memory
indep = np.empty((n_rows, 6144), dtype=np.float64)
csz = 1024
for c in range(0, n_rows, csz):
    e = min(c + csz, n_rows)
    raw = read_rows(mm, data_start, tinfo, np.arange(c, e), row_bytes)
    indep[c:e] = dequant_rows(raw)
print(f"independent dequant: {indep.shape} max|.|={np.abs(indep).max():.4f}")

# ---- engine q6k_gather dequant (w/*.pt) ----
parts, c = [], 0
while True:
    p = W / f"L03_attn_output_c{c}.pt"
    if not p.exists():
        break
    parts.append(torch.load(p, map_location="cpu"))
    c += 1
eng_dq = torch.cat(parts, dim=0).float().numpy().astype(np.float64)
print(f"engine q6k_gather dequant: {eng_dq.shape} max|.|={np.abs(eng_dq).max():.4f}")

print("\n=== dequant cross-check: independent f64 vs engine q6k_gather ===")
cmp("L03_attn_output", indep, eng_dq)

# ---- recompute f64 out-proj with BOTH dequants, compare to engine layer_03_r ----
ref = np.load(OUT / "ref_l3_inter.npz")
o_ref = ref["o"].astype(np.float64)
caps = torch.load(OUT / "captures.pt", map_location="cpu")
r2 = caps["layer_02_r"].float().numpy().astype(np.float64)
h2 = caps["layer_02_h"].float().numpy().astype(np.float64)
eng_r3 = caps["layer_03_r"].float().numpy().astype(np.float64)

a_indep = o_ref.reshape(o_ref.shape[0], -1) @ indep.T
a_engdq = o_ref.reshape(o_ref.shape[0], -1) @ eng_dq.T
print("\n=== f64 out-proj (indep dequant) vs engine layer_03_r ===")
cmp("r(indep_dq)", (r2 + h2) + a_indep, eng_r3)
print("=== f64 out-proj (engine-dq) vs engine layer_03_r ===")
cmp("r(eng_dq)", (r2 + h2) + a_engdq, eng_r3)

# ---- where do the two dequants differ? ----
d = np.abs(indep - eng_dq)
print(f"\ndequant self-diff: max={d.max():.5f} mean={d.mean():.5f} frac>0.01={float((d>0.01).mean()):.4f}")
