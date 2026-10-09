"""Independent verification of the ssm_conv1d GGUF alignment fix.

Parses the GGUF header/metadata/tensor-infos by hand (no shared code with the
engine reader), computes both the OLD (unaligned) and NEW (aligned) tensor-data
section starts, reads blk.0.ssm_conv1d.weight at each, and reports value stats.
Then cross-checks against the engine's GgufReader (the fixed path).
"""
import mmap
import struct
import sys
from pathlib import Path

import torch

GGUF = Path(
    "/home/hhy/Models/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF/"
    "Qwen3.8-27B-EfficientThink-SimPO-Q6_K.gguf"
)
TARGET = "blk.0.ssm_conv1d.weight"


def read_str(buf, pos):
    n = struct.unpack_from("<Q", buf, pos)[0]
    pos += 8
    s = buf[pos : pos + n].decode("utf-8")
    return s, pos + n


def read_value(buf, pos, vtype):
    if vtype == 0:
        return buf[pos], pos + 1
    if vtype == 1:
        return struct.unpack_from("<b", buf, pos)[0], pos + 1
    if vtype == 2:
        return struct.unpack_from("<H", buf, pos)[0], pos + 2
    if vtype == 3:
        return struct.unpack_from("<h", buf, pos)[0], pos + 2
    if vtype == 4:
        return struct.unpack_from("<I", buf, pos)[0], pos + 4
    if vtype == 5:
        return struct.unpack_from("<i", buf, pos)[0], pos + 4
    if vtype == 6:
        return struct.unpack_from("<f", buf, pos)[0], pos + 4
    if vtype == 7:
        return buf[pos] != 0, pos + 1
    if vtype == 8:
        s, pos = read_str(buf, pos)
        return s, pos
    if vtype == 9:
        et = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        cnt = struct.unpack_from("<Q", buf, pos)[0]
        pos += 8
        out = []
        for _ in range(cnt):
            v, pos = read_value(buf, pos, et)
            out.append(v)
        return out, pos
    if vtype == 10:
        return struct.unpack_from("<Q", buf, pos)[0], pos + 8
    if vtype == 11:
        return struct.unpack_from("<q", buf, pos)[0], pos + 8
    if vtype == 12:
        return struct.unpack_from("<d", buf, pos)[0], pos + 8
    raise ValueError(f"vtype {vtype}")


def stats(t: torch.Tensor, label: str) -> None:
    tf = t.float()
    nz = int((tf == 0).sum())
    finite = bool(torch.isfinite(tf).all())
    print(
        f"  {label:12s} min={tf.min():+.6f} max={tf.max():+.6f} "
        f"mean={tf.mean():+.6f} std={tf.std():.6f} zeros={nz}/{tf.numel()} finite={finite}"
    )
    print(f"               first8={tf.flatten()[:8].tolist()}")


def main() -> None:
    fh = open(GGUF, "rb")
    buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    pos = 0
    magic = struct.unpack_from("<I", buf, pos)[0]
    assert magic == 0x46554747, f"bad magic {magic:#x}"
    pos += 4
    version = struct.unpack_from("<I", buf, pos)[0]
    pos += 4
    n_tensors = struct.unpack_from("<Q", buf, pos)[0]
    pos += 8
    n_kv = struct.unpack_from("<Q", buf, pos)[0]
    pos += 8
    print(f"version={version} n_tensors={n_tensors} n_kv={n_kv}")

    alignment = 32
    for _ in range(n_kv):
        key, pos = read_str(buf, pos)
        vt = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        val, pos = read_value(buf, pos, vt)
        if key == "general.alignment":
            alignment = int(val)
    print(f"alignment={alignment}  (pos after metadata = {pos})")

    tensors = []
    for _ in range(n_tensors):
        name, pos = read_str(buf, pos)
        ndim = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        dims = list(struct.unpack_from(f"<{ndim}Q", buf, pos))
        pos += 8 * ndim
        ttype = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        offset = struct.unpack_from("<Q", buf, pos)[0]
        pos += 8
        tensors.append((name, ttype, dims[::-1], offset))

    unaligned_start = pos
    aligned_start = (pos + alignment - 1) // alignment * alignment
    print(f"pos after tensor-infos = {unaligned_start}")
    print(f"  OLD (unaligned) data start = {unaligned_start}")
    print(f"  NEW (aligned)   data start = {aligned_start}  (delta={aligned_start - unaligned_start})")

    # sanity: first tensor offset should be 0 (data section start == first tensor)
    print(f"  tensor[0] = {tensors[0][0]} offset={tensors[0][3]}")

    tgt = next((t for t in tensors if t[0] == TARGET), None)
    assert tgt is not None, f"{TARGET} not found"
    name, ttype, shape, offset = tgt
    print(f"\n{TARGET}: type={ttype} shape={shape} info.offset={offset}")
    assert ttype == 0, f"expected F32 (0), got {ttype}"
    n_bytes = 4 * int(torch.tensor(shape).prod())
    print(f"  expected bytes={n_bytes}")

    def read_at(base: int) -> torch.Tensor:
        start = base + offset
        raw = buf[start : start + n_bytes]
        assert len(raw) == n_bytes, f"short read at {start}"
        return torch.frombuffer(raw, dtype=torch.uint8).view(torch.float32).view(shape)

    print("\n  OLD reader (unaligned start):")
    stats(read_at(unaligned_start), "old")
    print("\n  NEW reader (aligned start):")
    new = read_at(aligned_start)
    stats(new, "new")

    # Cross-check against the engine's fixed GgufReader.
    sys.path.insert(0, str(Path(__file__).parents[1] / "python"))
    from minisgl.models.gguf_reader import GgufReader

    with GgufReader(GGUF) as r:
        info = next(i for i in r.tensors if i.name == TARGET)
        eng = r.tensor_torch(info)
    diff = (eng.float() - new.float()).abs().max().item()
    print(f"\n  engine GgufReader vs independent aligned read: max|diff| = {diff:.3e}")
    print("  MATCH" if diff == 0.0 else "  MISMATCH")


if __name__ == "__main__":
    main()
