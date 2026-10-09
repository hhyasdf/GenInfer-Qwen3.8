"""B4 bisection — CPU side (no GPU).

1. Independent GGUF parser (mmap + struct, written from the format spec,
   NOT from gguf_reader.py) → cross-check tensor geometry against the
   engine reader's view (reader_geom.json).
2. Independent numpy f64 Q6_K dequantizer (vectorized port of the
   kernel-validated f64_dequant_row in verify_q6k.py) → cross-check the
   engine's q6k_gather dequant (w/*.pt) on sampled rows.
3. f64 full-model reference (embedding → 64 layers → final norm → lm_head)
   → compare per-layer against the engine's captures (captures.pt),
   report the first divergence (relative error > 0.1, same threshold as
   the oracle).
4. lm_head bisection: engine logits vs f64(reference hidden) vs
   f64(engine hidden) — separates an upstream bug from an lm_head GEMV bug.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]  # engine root (this script lives in hacks/)
OUT = Path("/tmp/bisect")
W = OUT / "w"
CHUNK = 16384  # must match bisect_prefill.py
TOL_REL = 0.1  # per-layer divergence threshold (same as the oracle)
GOLDEN_TOL = 0.01  # golden logits tolerance (max abs diff)

# ---------------------------------------------------------------------------
# 1. Independent GGUF parser
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)


def _read_str(mm: mmap.mmap, pos: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<Q", mm, pos)
    pos += 8
    s = mm[pos : pos + n].decode("utf-8")
    pos += n
    return s, pos


def _read_raw(mm: mmap.mmap, pos: int, t: int):
    """Read the payload of a value of known type t (no type prefix).
    Strings still carry their own u64 length prefix."""
    if t == T_U8:
        return struct.unpack_from("<B", mm, pos)[0], pos + 1
    if t == T_I8:
        return struct.unpack_from("<b", mm, pos)[0], pos + 1
    if t == T_U16:
        return struct.unpack_from("<H", mm, pos)[0], pos + 2
    if t == T_I16:
        return struct.unpack_from("<h", mm, pos)[0], pos + 2
    if t == T_U32:
        return struct.unpack_from("<I", mm, pos)[0], pos + 4
    if t == T_I32:
        return struct.unpack_from("<i", mm, pos)[0], pos + 4
    if t == T_F32:
        return struct.unpack_from("<f", mm, pos)[0], pos + 4
    if t == T_BOOL:
        return struct.unpack_from("<?", mm, pos)[0], pos + 1
    if t == T_STR:
        return _read_str(mm, pos)
    if t == T_U64:
        return struct.unpack_from("<Q", mm, pos)[0], pos + 8
    if t == T_I64:
        return struct.unpack_from("<q", mm, pos)[0], pos + 8
    if t == T_F64:
        return struct.unpack_from("<d", mm, pos)[0], pos + 8
    raise ValueError(f"unknown GGUF value type {t}")


def _read_val(mm: mmap.mmap, pos: int):
    # GGUF spec (gguf.h): "All enums are stored as int32_t."
    (t,) = struct.unpack_from("<I", mm, pos)
    pos += 4
    if t == T_ARR:
        (et,) = struct.unpack_from("<I", mm, pos)  # element type, int32_t per spec
        pos += 4
        (cnt,) = struct.unpack_from("<Q", mm, pos)
        pos += 8
        vals = []
        for _ in range(cnt):
            v, pos = _read_raw(mm, pos, et)  # no per-element type prefix
            vals.append(v)
        return vals, pos
    return _read_raw(mm, pos, t)


def parse_gguf(path: Path):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    pos = 0
    magic, version, n_tensors, n_kv = struct.unpack_from("<IIQQ", mm, pos)
    pos += 24
    assert magic == GGUF_MAGIC, f"bad magic {magic:#x}"
    meta: dict = {}
    for _ in range(n_kv):
        key, pos = _read_str(mm, pos)
        val, pos = _read_val(mm, pos)
        meta[key] = val
    tensors = []
    for _ in range(n_tensors):
        name, pos = _read_str(mm, pos)
        (ndim,) = struct.unpack_from("<I", mm, pos)
        pos += 4
        dims = list(struct.unpack_from(f"<{ndim}Q", mm, pos))
        pos += 8 * ndim
        (ttype,) = struct.unpack_from("<I", mm, pos)
        pos += 4
        (offset,) = struct.unpack_from("<Q", mm, pos)
        pos += 8
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})
    align = int(meta.get("general.alignment", 32))
    data_start = (pos + align - 1) // align * align
    return meta, tensors, data_start, mm


# ---------------------------------------------------------------------------
# 2. Independent numpy f64 Q6_K dequantizer
# ---------------------------------------------------------------------------

# Per-position index arrays for one 256-element Q6_K block, derived from the
# kernel-validated f64_dequant_row in verify_q6k.py:
#   ql[128] @0, qh[64] @128, scales[16] int8 @192, d half @208
_POS = np.arange(256)
_N = (_POS // 128).astype(np.intp)
_L = (_POS % 128) % 32
_SUB = (_POS % 128) // 32
_IS = _L // 16
_QL_IDX = _N * 64 + _L + np.where(_SUB & 1, 32, 0)
_USE_HI = _SUB >= 2
_QH_IDX = _N * 32 + _L
_QH_SHIFT = _SUB * 2
_SC_IDX = _N * 8 + _IS + _SUB * 2


def dequant_rows(rows_u8: np.ndarray) -> np.ndarray:
    """rows_u8: [R, nb*210] uint8 → [R, nb*256] f64."""
    R, total = rows_u8.shape
    nb = total // 210
    rows_u8 = rows_u8.reshape(R, nb, 210)
    ql = rows_u8[:, :, :128]
    qh = rows_u8[:, :, 128:192]
    sc = rows_u8[:, :, 192:208].astype(np.int8).astype(np.int16)
    d = (
        np.frombuffer(rows_u8[:, :, 208:210].tobytes(), dtype=np.float16)
        .reshape(R, nb)
        .astype(np.float64)
    )
    qlv = ql[:, :, _QL_IDX]
    low = np.where(_USE_HI[None, None, :], qlv >> 4, qlv & 0xF)
    qhv = qh[:, :, _QH_IDX]
    high = (qhv >> _QH_SHIFT[None, None, :]) & 3
    scv = sc[:, :, _SC_IDX]
    q = (low | (high << 4)).astype(np.int16) - 32  # signed mantissa, range -32..31
    return (d[:, :, None] * scv.astype(np.float64) * q.astype(np.float64)).reshape(R, nb * 256)


def read_rows(mm, data_start, tinfo, rows, row_bytes):
    out = np.empty((len(rows), row_bytes), dtype=np.uint8)
    for i, r in enumerate(rows):
        off = data_start + tinfo["offset"] + int(r) * row_bytes
        out[i] = np.frombuffer(mm[off : off + row_bytes], dtype=np.uint8)
    return out


# ---------------------------------------------------------------------------
# 3. f64 model reference
# ---------------------------------------------------------------------------


def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    # matches torch F.softplus (beta=1, threshold=20)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def ref_gdn_attn(x, W, eps, state):
    n = x.shape[0]
    qkv_mixed = x @ W["attn_qkv"].T  # [n, 10240]
    z = x @ W["attn_gate"].T  # [n, 6144]
    beta = sigmoid(x @ W["ssm_beta"].T)  # [n, 48]
    alpha = x @ W["ssm_alpha"].T  # [n, 48]
    # GDN gate: ssm_a * softplus(alpha + dt).  The GGUF stores the
    # pre-transformed -exp(A_log) (conversion script applies -torch.exp),
    # so this equals HF's g = -A_log.exp()*softplus.
    gate = softplus(alpha + W["ssm_dt"]) * W["ssm_a"]  # [n, 48]
    # conv1d kernel 4 + silu (conv_in = [conv_buf(3), qkv_mixed(n)])
    conv_in = np.concatenate([state["conv_buf"], qkv_mixed], axis=0)
    w = W["ssm_conv1d"]  # [10240, 4]
    conv_out = (
        w[:, 0][None, :] * conv_in[0:n]
        + w[:, 1][None, :] * conv_in[1 : n + 1]
        + w[:, 2][None, :] * conv_in[2 : n + 2]
        + w[:, 3][None, :] * conv_in[3 : n + 3]
    )
    conv_out = silu(conv_out)
    state["conv_buf"] = conv_in[-3:].copy()
    # split + L2 normalize (eps 1e-6)
    q = conv_out[:, :2048].reshape(n, 16, 128)
    k = conv_out[:, 2048:4096].reshape(n, 16, 128)
    v = conv_out[:, 4096:].reshape(n, 48, 128)
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    # recurrence, kernel's transposed state storage: S[h, i, col]
    S = state["S"]  # [48, 128, 128]
    scale = 1.0 / np.sqrt(128.0)
    o = np.empty((n, 48, 128))
    for t in range(n):
        for h in range(48):
            qk = h % 16
            g = np.exp(gate[t, h])
            kv = S[h].T @ k[t, qk]  # [128]
            delta = (v[t, h] - g * kv) * beta[t, h]
            S[h] = g * S[h] + np.outer(k[t, qk], delta)
            o[t, h] = (S[h].T @ q[t, qk]) * scale
    # gated output
    o = o.reshape(-1, 128)
    o = rmsnorm(o, W["ssm_norm"], eps)
    o = o.reshape(n, 6144) * silu(z)
    return o @ W["ssm_out"].T  # [n, 5120]


def ref_fullattn_attn(x, W, eps, positions, rope_base, rope_dim):
    n = x.shape[0]
    qg = x @ W["attn_q"].T  # [n, 12288]
    # per-head interleaved split (llama.cpp qwen35.cpp): each 512-dim head =
    # [256 Q | 256 gate]; head-stride 512, gate offset 256. NOT a global split.
    qg = qg.reshape(n, 24, 512)
    q = qg[:, :, :256]
    gate = qg[:, :, 256:]
    q = rmsnorm(q, W["attn_q_norm"], eps)
    k = (x @ W["attn_k"].T).reshape(n, 4, 256)
    k = rmsnorm(k, W["attn_k_norm"], eps)
    v = (x @ W["attn_v"].T).reshape(n, 4, 256)
    # partial RoPE: rope_dim/2 = 32 pairs over the first rope_dim = 64 dims
    half = rope_dim // 2
    theta = rope_base ** (-2.0 * np.arange(half) / rope_dim)
    ang = positions[:, None] * theta[None, :]
    cos = np.cos(ang)[:, None, :]
    sin = np.sin(ang)[:, None, :]

    def rope(xh):
        x1 = xh[..., :half].copy()
        x2 = xh[..., half : 2 * half].copy()
        out = xh.copy()
        out[..., :half] = x1 * cos - x2 * sin
        out[..., half : 2 * half] = x2 * cos + x1 * sin
        return out

    q = rope(q)
    k = rope(k)
    # causal GQA attention, scale 1/sqrt(256) = 1/16, kv head = q head // 6
    k_e = np.repeat(k, 6, axis=1)  # [n, 24, 256]
    v_e = np.repeat(v, 6, axis=1)
    scores = np.einsum("thd,shd->ths", q, k_e) / 16.0
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(mask[:, None, :], -1e30, scores)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    o = np.einsum("ths,shd->thd", p, v_e)  # [n, 24, 256]
    o = o * sigmoid(gate)
    return o.reshape(n, 6144) @ W["attn_output"].T  # [n, 5120]


def ref_decoder_layer(x, residual, W, eps, kind, state=None, positions=None,
                      rope_base=1e7, rope_dim=64):
    if residual is None:
        residual = x
        x_norm = rmsnorm(x, W["input_layernorm"], eps)
    else:
        residual = residual + x
        x_norm = rmsnorm(residual, W["input_layernorm"], eps)
    if kind == "gdn":
        a = ref_gdn_attn(x_norm, W, eps, state)
    else:
        a = ref_fullattn_attn(x_norm, W, eps, positions, rope_base, rope_dim)
    residual = residual + a
    x_norm2 = rmsnorm(residual, W["post_attention_layernorm"], eps)
    g = x_norm2 @ W["mlp_gate"].T
    u = x_norm2 @ W["mlp_up"].T
    h = (silu(g) * u) @ W["mlp_down"].T
    return h, residual


# ---------------------------------------------------------------------------
# 4. Comparison
# ---------------------------------------------------------------------------


def compare(name, ref, eng, tol_rel=TOL_REL):
    eng = np.asarray(eng, dtype=np.float64)
    diff = np.abs(ref - eng)
    max_diff = float(diff.max())
    scale = max(float(np.abs(ref).max()), 1e-9)
    rel = max_diff / scale
    frac = float((diff > 0.1 * np.abs(ref)).mean())
    ok = rel <= tol_rel and frac < 0.05
    print(
        f"{'OK     ' if ok else 'DIVERGE'} {name:16s} max|diff|={max_diff:.4e}  "
        f"rel={rel:.3e}  frac(|d|>0.1|ref|)={frac:.4f}"
    )
    return ok, max_diff, rel


def load_w(name):
    parts = []
    c = 0
    while True:
        p = W / f"{name}_c{c}.pt"
        if not p.exists():
            break
        parts.append(torch.load(p, map_location="cpu"))
        c += 1
    assert parts, f"missing weight file {name}"
    return torch.cat(parts, dim=0).float().numpy().astype(np.float64)


def load_layer_weights(i, kind, small):
    s = small[f"L{i:02d}"]
    W = {
        "mlp_gate": load_w(f"L{i:02d}_mlp_gate"),
        "mlp_up": load_w(f"L{i:02d}_mlp_up"),
        "mlp_down": load_w(f"L{i:02d}_mlp_down"),
        "input_layernorm": s["input_layernorm"].float().numpy().astype(np.float64),
        "post_attention_layernorm": s["post_attention_layernorm"].float().numpy().astype(np.float64),
    }
    if kind == "gdn":
        W["attn_qkv"] = load_w(f"L{i:02d}_attn_qkv")
        W["attn_gate"] = load_w(f"L{i:02d}_attn_gate")
        W["ssm_beta"] = load_w(f"L{i:02d}_ssm_beta")
        W["ssm_alpha"] = load_w(f"L{i:02d}_ssm_alpha")
        W["ssm_out"] = load_w(f"L{i:02d}_ssm_out")
        W["ssm_conv1d"] = s["ssm_conv1d"].float().numpy().astype(np.float64)
        W["ssm_dt"] = s["ssm_dt"].float().numpy().astype(np.float64)
        W["ssm_a"] = s["ssm_a"].float().numpy().astype(np.float64)
        W["ssm_norm"] = s["ssm_norm"].float().numpy().astype(np.float64)
    else:
        W["attn_q"] = load_w(f"L{i:02d}_attn_q")
        W["attn_k"] = load_w(f"L{i:02d}_attn_k")
        W["attn_v"] = load_w(f"L{i:02d}_attn_v")
        W["attn_output"] = load_w(f"L{i:02d}_attn_output")
        W["attn_q_norm"] = s["attn_q_norm"].float().numpy().astype(np.float64)
        W["attn_k_norm"] = s["attn_k_norm"].float().numpy().astype(np.float64)
    return W


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    meta = json.loads((OUT / "meta.json").read_text())
    caps = torch.load(OUT / "captures.pt", map_location="cpu")
    small = torch.load(OUT / "small.pt", map_location="cpu")
    reader_geom = json.loads((OUT / "reader_geom.json").read_text())

    import sys
    import tomllib

    sys.path.insert(0, str(ROOT / "python"))
    from minisgl.models.gguf_weight import find_main_gguf

    cfg = tomllib.load(open(ROOT / "config.toml", "rb"))
    model_path = cfg["engine"]["model"]
    gguf_path = find_main_gguf(model_path)
    print(f"GGUF: {gguf_path}")

    # ---- 1. independent parser + geometry cross-check ----
    gmeta, gtensors, data_start, mm = parse_gguf(gguf_path)
    print(f"parsed: {len(gtensors)} tensors, {len(gmeta)} metadata keys")
    gmap = {t["name"]: t for t in gtensors}

    mism = 0
    for name, g in reader_geom.items():
        t = gmap.get(name)
        if t is None:
            print(f"GEOM-MISSING in independent parser: {name}")
            mism += 1
            continue
        if list(reversed(t["dims"])) != g["shape"]:
            print(f"GEOM-SHAPE {name}: independent {t['dims']} vs reader {g['shape']}")
            mism += 1
        if t["type"] != g["type"]:
            print(f"GEOM-TYPE {name}: independent {t['type']} vs reader {g['type']}")
            mism += 1
        if t["offset"] != g["offset"]:
            print(f"GEOM-OFFSET {name}: independent {t['offset']} vs reader {g['offset']}")
            mism += 1
    for name in gmap:
        if name not in reader_geom:
            print(f"GEOM-EXTRA in independent parser: {name}")
            mism += 1
    print(f"geometry cross-check: {'PASS' if mism == 0 else f'FAIL ({mism} mismatches)'}")

    # GGUF metadata is namespaced under the architecture prefix
    # (general.architecture + "."), exactly like the engine's from_gguf.
    pfx = str(gmeta.get("general.architecture", "")) + "."

    def g(key: str, default=None):
        return gmeta.get(pfx + key, default)

    eps = float(g("attention.layer_norm_rms_epsilon", 1e-6))
    rope_base = float(g("rope.freq_base", 1e7))
    sections = g("rope.dimension_sections", [])
    # The engine reads rope.dimension_count (64): the number of rotated dims
    # per head (32 frequency pairs over the first 64 dims, pairs (i, i+32)).
    # Fallback: 2 * sum(sections) = 2 * 32 = 64.
    rope_dim = int(g("rope.dimension_count", int(2 * sum(sections)) if sections else 64))
    print(
        f"metadata: eps={eps} rope_base={rope_base} sections={sections} rope_dim={rope_dim} "
        f"(engine meta eps={meta['eps']})"
    )
    assert abs(eps - meta["eps"]) < 1e-12, "eps mismatch between GGUF and engine config"

    # ---- 2. dequant cross-check on sampled rows ----
    print("\n--- dequant cross-check (independent numpy f64 vs engine kernel bf16) ---")

    def crosscheck(wname, gguf_name, n_rows):
        tinfo = gmap[gguf_name]
        row_bytes = tinfo["dims"][0] // 256 * 210
        rows = np.unique(np.linspace(0, n_rows - 1, 32)).astype(np.int64)
        raw = read_rows(mm, data_start, tinfo, rows, row_bytes)
        ref = dequant_rows(raw)  # [32, K] f64
        eng = []
        for r in rows:
            c = int(r) // CHUNK
            part = torch.load(W / f"{wname}_c{c}.pt", map_location="cpu")
            eng.append(part[int(r) % CHUNK].float().numpy().astype(np.float64))
        eng = np.stack(eng)
        ok, md, rel = compare(f"{wname}", ref, eng)
        return ok

    dq_ok = all(
        [
            crosscheck("embed", "token_embd.weight", 248320),
            crosscheck("lm_head", "output.weight", 248320),
            crosscheck("L00_attn_qkv", "blk.0.attn_qkv.weight", 10240),
            crosscheck("L03_attn_q", "blk.3.attn_q.weight", 12288),
        ]
    )

    # ---- 3. f64 full-model reference ----
    print("\n--- f64 full-model reference vs engine captures ---")
    ids = np.array(meta["prompt_ids"], dtype=np.int64)
    n = len(ids)
    positions = np.arange(n)

    # embedding: independent numpy dequant of the prompt rows
    tinfo = gmap["token_embd.weight"]
    row_bytes = tinfo["dims"][0] // 256 * 210
    embed_ref = dequant_rows(read_rows(mm, data_start, tinfo, ids, row_bytes))  # [73, 5120]
    ok_embed, _, _ = compare("embed", embed_ref, caps["embed"].float().numpy())

    first_div = None
    h = embed_ref
    residual = None
    for i in range(64):
        kind = meta["layer_kinds"][i]
        t = time.time()
        lw = load_layer_weights(i, kind, small)
        if kind == "gdn":
            state = {"S": np.zeros((48, 128, 128)), "conv_buf": np.zeros((3, 10240))}
            h, residual = ref_decoder_layer(h, residual, lw, eps, kind, state)
        else:
            h, residual = ref_decoder_layer(
                h, residual, lw, eps, kind, positions=positions,
                rope_base=rope_base, rope_dim=rope_dim,
            )
        dt = time.time() - t
        ok_h, _, _ = compare(f"layer_{i:02d}_h", h, caps[f"layer_{i:02d}_h"].float().numpy())
        ok_r, _, _ = compare(f"layer_{i:02d}_r", residual, caps[f"layer_{i:02d}_r"].float().numpy())
        if not (ok_h and ok_r) and first_div is None:
            first_div = i
            print(f"*** FIRST DIVERGENCE at layer {i} ({kind}) ***")
        print(f"  (layer {i} {kind}: {dt:.1f} s, {time.time() - t0:.0f} s total)")

    final_ref = rmsnorm(h + residual, small["final_norm"].float().numpy().astype(np.float64), eps)
    ok_final, _, _ = compare("final_hidden", final_ref, caps["final_hidden"].float().numpy())

    # ---- 4. lm_head bisection ----
    print("\n--- lm_head bisection ---")
    tinfo = gmap["output.weight"]
    row_bytes = tinfo["dims"][0] // 256 * 210
    eng_logits = caps["logits"].float().numpy().astype(np.float64)  # [248320]

    def lm_gemv(hidden_row):
        logits = np.empty(248320)
        for c in range(0, 248320, 8192):
            e = min(c + 8192, 248320)
            raw = read_rows(mm, data_start, tinfo, np.arange(c, e), row_bytes)
            lm = dequant_rows(raw)  # [chunk, 5120] f64
            logits[c:e] = lm @ hidden_row
        return logits

    logits_from_ref_hidden = lm_gemv(final_ref[-1])
    ok_lr, md_lr, rel_lr = compare("logits(ref hid)", logits_from_ref_hidden, eng_logits)
    print(f"  golden tolerance: max|diff|={md_lr:.4e} vs {GOLDEN_TOL} → "
          f"{'PASS' if md_lr <= GOLDEN_TOL else 'FAIL'}")

    # Save the f64 reference logits and compare to the baked golden reference
    # (independent oracle) — this is the ~31.6 error measurement.
    np.save(OUT / "f64_logits.npy", logits_from_ref_hidden)
    try:
        import json as _json
        _g = _json.load(open(ROOT / "golden" / "logits.json"))
        _golden_logits = np.array(_g["prompts"][0]["logits"], dtype=np.float64)
        _diff = logits_from_ref_hidden - _golden_logits
        print(f"  f64-ref vs GOLDEN oracle: max|diff|={np.abs(_diff).max():.4e} "
              f"mean|diff|={np.abs(_diff).mean():.4e} "
              f"top1_ref={int(np.argmax(logits_from_ref_hidden))} "
              f"top1_golden={int(np.argmax(_golden_logits))}")
    except Exception as _e:
        print(f"  (golden compare skipped: {_e})")

    eng_final = caps["final_hidden"].float().numpy().astype(np.float64)[-1]
    logits_from_eng_hidden = lm_gemv(eng_final)
    ok_le, md_le, rel_le = compare("logits(eng hid)", logits_from_eng_hidden, eng_logits)
    print(f"  golden tolerance: max|diff|={md_le:.4e} vs {GOLDEN_TOL} → "
          f"{'PASS' if md_le <= GOLDEN_TOL else 'FAIL'}")

    # ---- verdict ----
    print("\n--- verdict ---")
    print(f"geometry: {'PASS' if mism == 0 else 'FAIL'}")
    print(f"dequant cross-check: {'PASS' if dq_ok else 'FAIL'}")
    print(f"first layer divergence: {first_div}")
    print(f"logits vs f64(ref hidden):  max|diff|={md_lr:.4e}")
    print(f"logits vs f64(eng hidden):  max|diff|={md_le:.4e}")
    if md_le <= GOLDEN_TOL and md_lr > GOLDEN_TOL:
        print("→ lm_head GEMV is CORRECT; the bug is UPSTREAM (a layer).")
    elif md_lr <= GOLDEN_TOL:
        print("→ prefill is CORRECT end-to-end; the selftest failure is elsewhere.")
    elif first_div is not None:
        print(f"→ bug localizes to layer {first_div} or earlier.")
    else:
        print("→ logits wrong but every layer matches: check final norm / lm_head path.")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
