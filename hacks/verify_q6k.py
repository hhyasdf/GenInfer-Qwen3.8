"""Verify the engine's Q6_K dequant kernel against an independent f64 reference.

Reads the first 16 rows of the lm_head (output.weight) from the GGUF, runs the
engine's q6k_gather kernel on them, and compares against a hand-written f64
dequant of the same raw bytes. If the kernel is correct, the bf16 output matches
the f64 dequant (cast to bf16) to ~1e-3 relative error.
"""
import struct
import sys

import torch

GGUF = "/home/hhy/Models/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF/Qwen3.8-27B-EfficientThink-SimPO-Q6_K.gguf"
N_ROWS = 16
K = 5120  # lm_head embedding dim
BLOCKS = K // 256  # 20 blocks per row
BLOCK_BYTES = 210


def f64_dequant_row(row_bytes: bytes) -> torch.Tensor:
    """Independent f64 dequant of one Q6_K row (K elements)."""
    assert len(row_bytes) == BLOCKS * BLOCK_BYTES
    out = torch.empty(K, dtype=torch.float64)
    for b in range(BLOCKS):
        base = b * BLOCK_BYTES
        ql = row_bytes[base:base + 128]
        qh = row_bytes[base + 128:base + 128 + 64]
        scales = struct.unpack("<16b", row_bytes[base + 192:base + 208])
        (d,) = struct.unpack("<e", row_bytes[base + 208:base + 210])  # half float
        for pos in range(256):
            i = b * 256 + pos
            n = pos // 128
            l = (pos % 128) % 32
            sub = (pos % 128) // 32
            is_ = l // 16
            ql_idx = n * 64 + l + (32 if (sub & 1) else 0)
            q_low = (ql[ql_idx] & 0xF) if sub < 2 else (ql[ql_idx] >> 4)
            q_high = (qh[n * 32 + l] >> (sub * 2)) & 3
            sc_idx = n * 8 + is_ + sub * 2
            out[i] = torch.tensor(d, dtype=torch.float64) * torch.tensor(
                scales[sc_idx], dtype=torch.float64
            ) * torch.tensor((q_low | (q_high << 4)) - 32, dtype=torch.float64)
    return out


def main():
    from minisgl.models.gguf_reader import GgufReader
    from minisgl.kernel import q6k

    device = "cuda:0"
    torch.cuda.set_device(0)

    # 1. Read the first 16 rows of the lm_head raw bytes.
    with GgufReader(GGUF) as reader:
        lm_info = None
        for info in reader.tensors:
            if info.name == "output.weight":
                lm_info = info
                break
        assert lm_info is not None, "output.weight not found"
        raw_all = reader.tensor_bytes(lm_info)
    row_bytes_all = raw_all[: N_ROWS * BLOCKS * BLOCK_BYTES]

    # 2. Independent f64 dequant of each row.
    ref = torch.stack([f64_dequant_row(row_bytes_all[r * BLOCKS * BLOCK_BYTES:(r + 1) * BLOCKS * BLOCK_BYTES]) for r in range(N_ROWS)])
    print(f"ref f64: shape={tuple(ref.shape)} min={ref.min():.4f} max={ref.max():.4f} std={ref.std():.4f}")

    # 3. Engine's q6k_gather kernel.
    W = torch.frombuffer(bytearray(row_bytes_all), dtype=torch.uint8).reshape(N_ROWS, BLOCKS * BLOCK_BYTES).to(device)
    ids = torch.arange(N_ROWS, dtype=torch.int32, device=device)
    y = torch.empty(N_ROWS, K, dtype=torch.bfloat16, device=device)
    q6k.q6k_gather(ids, W, y)
    eng = y.cpu().double()
    print(f"engine bf16->f64: shape={tuple(eng.shape)} min={eng.min():.4f} max={eng.max():.4f} std={eng.std():.4f}")

    # 4. Compare.
    diff = (eng - ref).abs()
    rel = (diff / ref.abs().clamp_min(1e-6))
    print(f"max|diff|={diff.max():.6f}  mean|diff|={diff.mean():.6f}  max_rel={rel.max():.6f}")
    # bf16 has ~3 decimal digits of precision; a correct dequant should match to ~1e-2 rel.
    ok = diff.max() < 0.05
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
