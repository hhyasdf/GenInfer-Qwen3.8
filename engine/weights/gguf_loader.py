# Phase 7 (PP variant) — GGUF reader + dequant (weight_loading_contracts.md)。
#
# MODEL_DIR 是 GGUF-only (无 safetensors index):
#   Qwen3.8-27B-...-Q6_K-MTP.gguf (主模型) + mmproj-...-Q8_0.gguf (视觉) + config.json。
# 量化: Q6_K (主权重), Q8_0 (token_embd/output/MTP), BF16 (ssm alpha/beta), F32 (norms)。
#
# Round-2 D-6a 核心修复 — 量化驻留 VRAM, 逐 GEMM 反量化:
#   Round-1 在 host 把全部权重反量化成 bf16 再上卡 → rank1 (小显存卡) host OOM SIGKILL。
#   现在: 量化字节保持驻留 VRAM (uint8 buffer), 每次 GEMM 前对该 tensor 反量化。
#   Q6_K 驻留 = 0.5625 bytes/elem, 远小于 bf16 的 2.0。token_embd 用行 gather 反量化。
#
# GGUF 格式: magic "GGUF" + version(u32) + n_tensors(u64) + n_kv(u64) + metadata + tensor info + data。
# Q6_K dequant 对照 llama.cpp ggml-quants.c:1939-1968 (本 checkout 无 dmin: y = d·scales[j]·(q−32))。
# Q8_0 dequant 对照 ggml-quants.c:553-567 (y = d·qs)。
import struct
import numpy as np
import torch

# GGML tensor types (对照 llama.cpp ggml.h:389-434 类型枚举)
GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8
GGML_Q6_K = 14
GGML_BF16 = 30

# Q6_K block layout (210 bytes, ggml-common.h block_q6_K): ql(128) + qh(64) + scales(16) + d(f16,2)
Q6_K_BLOCK = 210
QK_K = 256
# Q8_0 block layout (34 bytes): d(f16,2) + qs(32) — 对照 ggml-quants.c:553-567 + GGUF offset 差值实测
Q8_0_BLOCK = 34


# ---------------------------------------------------------------------------
# numpy 反量化 (对照 llama.cpp C, 用于 CPU/bf16 合成前向冒烟 + 小 tensor)
# ---------------------------------------------------------------------------
def _dequant_q8_0(raw: bytes, n: int) -> np.ndarray:
    """Q8_0: 32/block, d(f16) + qs[32](int8)。y = d * qs。对照 ggml-quants.c:553-567。"""
    out = np.zeros(n, dtype=np.float32)
    nb = n // 32
    for i in range(nb):
        base = i * Q8_0_BLOCK
        # d 转 python float: 否则 float16 标量 × int8 在 f16 下运算溢出 inf/nan
        d = float(np.frombuffer(raw, dtype=np.float16, count=1, offset=base)[0])
        qs = np.frombuffer(raw, dtype=np.int8, count=32, offset=base + 2)
        out[i * 32:(i + 1) * 32] = d * qs
    return out


def _dequant_q6_K(raw: bytes, n: int) -> np.ndarray:
    """Q6_K: 256/block。对照 ggml-quants.c dequantize_row_q6_K (本 checkout 无 dmin)。"""
    out = np.zeros(n, dtype=np.float32)
    nb = n // QK_K
    for i in range(nb):
        base = i * Q6_K_BLOCK
        # 块布局 (ggml-common.h block_q6_K): ql[128](0) | qh[64](128) | scales[16](192) | d(f16)(208)
        # d 转 python float: 否则 float16 标量 × int8 在 f16 下运算溢出 inf/nan
        d = float(np.frombuffer(raw, dtype=np.float16, count=1, offset=base + 208)[0])
        ql = np.frombuffer(raw, dtype=np.uint8, count=128, offset=base)
        qh = np.frombuffer(raw, dtype=np.uint8, count=64, offset=base + 128)
        sc = np.frombuffer(raw, dtype=np.int8, count=16, offset=base + 192)
        for nn in range(QK_K // 128):
            # C 指针推进: 每 128 组 ql += 64, qh += 32, sc += 8
            qo, ho, so = 64 * nn, 32 * nn, 8 * nn
            for l in range(32):
                is_ = l // 16
                q1 = int((ql[l + qo] & 0xF) | (((qh[l + ho] >> 0) & 3) << 4)) - 32
                q2 = int((ql[l + 32 + qo] & 0xF) | (((qh[l + ho] >> 2) & 3) << 4)) - 32
                q3 = int((ql[l + qo] >> 4) | (((qh[l + ho] >> 4) & 3) << 4)) - 32
                q4 = int((ql[l + 32 + qo] >> 4) | (((qh[l + ho] >> 6) & 3) << 4)) - 32
                bo = i * QK_K
                out[bo + (nn * 128) + l + 0] = d * sc[is_ + so + 0] * q1
                out[bo + (nn * 128) + l + 32] = d * sc[is_ + so + 2] * q2
                out[bo + (nn * 128) + l + 64] = d * sc[is_ + so + 4] * q3
                out[bo + (nn * 128) + l + 96] = d * sc[is_ + so + 6] * q4
    return out


def _dequant(raw: bytes, n: int, t: int) -> np.ndarray:
    """把 GGUF raw 字节反量化成 f32 numpy。仅 F32/F16/BF16/Q8_0/Q6_K。"""
    if t == GGML_F32:
        return np.frombuffer(raw[:n * 4], dtype=np.float32).copy()
    if t == GGML_F16:
        return np.frombuffer(raw[:n * 2], dtype=np.float16).astype(np.float32)
    if t == GGML_BF16:
        u16 = np.frombuffer(raw[:n * 2], dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return f32
    if t == GGML_Q8_0:
        return _dequant_q8_0(raw, n)
    if t == GGML_Q6_K:
        return _dequant_q6_K(raw, n)
    raise ValueError(f"unsupported GGML tensor type {t}")


def _read_str(f) -> str:
    n = struct.unpack('<q', f.read(8))[0]
    return f.read(n).decode('utf-8')


def _read_value(f, vt: int):
    if vt == 0:   # uint8
        return struct.unpack('<B', f.read(1))[0]
    if vt == 1:   # int8
        return struct.unpack('<b', f.read(1))[0]
    if vt == 2:   # uint16
        return struct.unpack('<H', f.read(2))[0]
    if vt == 3:   # int16
        return struct.unpack('<h', f.read(2))[0]
    if vt == 4:   # uint32
        return struct.unpack('<I', f.read(4))[0]
    if vt == 5:   # int32
        return struct.unpack('<i', f.read(4))[0]
    if vt == 6:   # float32
        return struct.unpack('<f', f.read(4))[0]
    if vt == 7:   # bool
        return bool(struct.unpack('<b', f.read(1))[0])
    if vt == 8:   # string
        return _read_str(f)
    if vt == 9:   # array
        et = struct.unpack('<I', f.read(4))[0]
        cnt = struct.unpack('<q', f.read(8))[0]
        return [_read_value(f, et) for _ in range(cnt)]
    if vt == 10:  # uint64
        return struct.unpack('<Q', f.read(8))[0]
    if vt == 11:  # int64
        return struct.unpack('<q', f.read(8))[0]
    if vt == 12:  # float64
        return struct.unpack('<d', f.read(8))[0]
    raise ValueError(f"unsupported GGUF value type {vt}")


def read_gguf(path) -> tuple[dict, dict]:
    """读 GGUF 头 → (metadata, {tensor_name: (shape, ggml_type, data_offset)})。
    shape 是 GGUF 序 (innermost, ..., outermost), 调用方自行反转为 torch 序。"""
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', f"not a GGUF file: {path}"
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        metadata = {}
        for _ in range(n_kv):
            key = _read_str(f)
            vt = struct.unpack('<I', f.read(4))[0]
            metadata[key] = _read_value(f, vt)
        tensor_info = {}
        for _ in range(n_tensors):
            name = _read_str(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            t = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensor_info[name] = (tuple(dims), t, offset)
    return metadata, tensor_info


def tensor_nbytes(shape: tuple, t: int) -> int:
    """一个 GGUF tensor 的 raw 字节数 (按量化布局)。shape 是 GGUF 序。"""
    n = int(np.prod(shape)) if shape else 1
    if t == GGML_F32:
        return n * 4
    if t in (GGML_F16, GGML_BF16):
        return n * 2
    if t == GGML_Q8_0:
        return (n // 32) * Q8_0_BLOCK
    if t == GGML_Q6_K:
        return (n // QK_K) * Q6_K_BLOCK
    raise ValueError(f"unsupported type {t}")


def load_gguf_raw(path, names: list[str] | None = None,
                  device="cpu") -> dict[str, dict]:
    """读 GGUF → {name: {raw: torch.uint8, shape: torch序, ggml_type: int}}。
    量化字节原样驻留 (不上卡前不反量化) — D-6a 量化驻留路径。
    names=None 则全读。"""
    metadata, tensor_info = read_gguf(path)
    out: dict[str, dict] = {}
    with open(path, 'rb') as f:
        for name, (shape, t, offset) in tensor_info.items():
            if names is not None and name not in names:
                continue
            nbytes = tensor_nbytes(shape, t)
            f.seek(offset)
            raw = f.read(nbytes)
            arr = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            out[name] = {"raw": arr, "shape": tuple(reversed(shape)), "ggml_type": t}
    return out


def load_gguf_tensors(path, names: list[str] | None = None) -> dict:
    """反量化 GGUF tensors → {name: torch f32 tensor} (torch 序 shape)。
    用于 CPU/bf16 合成前向冒烟与小 tensor 预反量化 (非主权重路径)。"""
    metadata, tensor_info = read_gguf(path)
    out = {}
    with open(path, 'rb') as f:
        for name, (shape, t, offset) in tensor_info.items():
            if names is not None and name not in names:
                continue
            n = int(np.prod(shape)) if shape else 1
            nbytes = tensor_nbytes(shape, t)
            f.seek(offset)
            raw = f.read(nbytes)
            arr = _dequant(raw, n, t)
            torch_shape = list(reversed(shape))
            arr = arr.reshape(torch_shape)
            out[name] = torch.from_numpy(arr).clone()
    return out


def read_chat_template(path) -> str:
    """从 GGUF metadata 读 chat_template (tokenizer.ggml.* 内嵌)。"""
    metadata, _ = read_gguf(path)
    return metadata.get('tokenizer.chat_template', '')


def read_tokenizer_meta(path) -> dict:
    """从 GGUF metadata 抽 tokenizer.ggml.* → dict。供 D-6b 真实 BPE 用。
    返回 keys: tokens(list[str]), merges(list[list[str]]), pre(str), bos_token_id,
    eos_token_id, model(str), byte_encode(bool), special_tokens(dict)。"""
    metadata, _ = read_gguf(path)
    def g(key, default=None):
        return metadata.get(key, default)
    return {
        "tokens": g("tokenizer.ggml.tokens", []),
        "merges": g("tokenizer.ggml.merges", []),
        "pre": g("tokenizer.ggml.pre", ""),
        "model": g("tokenizer.ggml.model", "llama"),
        "bos_token_id": g("tokenizer.ggml.bos_token_id", -1),
        "eos_token_id": g("tokenizer.ggml.eos_token_id", -1),
        "byte_encode": g("tokenizer.ggml.use_byte_fallback", True),
        "special_tokens": {},
    }
