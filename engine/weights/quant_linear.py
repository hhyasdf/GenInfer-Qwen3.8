# D-6a — 量化驻留线性层 (Q6_K / Q8_0 torch 反量化)。
#
# Round-1 全量反量化到 host RAM (rank1 主机 dequant OOM SIGKILL) + bf16 全量驻留 → L3 零 token。
# 现在: GGUF 量化字节原样驻留 (uint8), 每次 GEMM 在设备现场反量化后立即释放。
#
# Round-3 构造期占位去全触碰 (L3 死因修复):
#   Round-2 的 __init__ 用 f32 torch.empty + kaiming_uniform_/normal_/zeros_ **全触碰** —
#   单层 GDN 实测 1.46GB × 两 rank ~99GB > 29GB RAM → 构造期 SIGKILL, 权重从未进显存。
#   现在: 占位 = torch.empty(..., device="meta") (对照 vllm gguf_loader.py:174 meta 哑模型)
#   — 零 host、零 VRAM、零触碰。真实分配只有两个点:
#     (a) load_quantized: raw 量化字节由 loader 直接放在目标设备 (VRAM 唯一分配点);
#     (b) 无 loader 前向 (self_check k6 合成冒烟): _weight/forward 首次触碰时
#         torch.empty(shape, dtype, device="cpu") 惰性物化 — 仍无全触碰 init。
#   ⛔ 本文件禁止 kaiming/normal_/zeros_/ones_ 任何全触碰 init。
#   - QuantLinear: forward 时 dequant(raw)→x.dtype 做 F.linear, 不保留完整 bf16 权重。
#     峰值 = 1 个 GEMM 的反量化权重 (FFN 17408×5120×2B ≈ 170MB), 对照 D-6a VRAM 预算。
#   - QuantEmbedding: token_embd Q8_0 [248320,5120] 按行 gather 后反量化 (T×5120×2B),
#     避免 2.5GB 全量反量化。
#   - 同时保留 float 路径 (load_dequantized) 供 CPU/bf16 合成前向冒烟。
#
# Q6_K 布局 (对照 ggml-common.h block_q6_K + ggml-quants.c dequantize_row_q6_K, 210 字节/块):
#   ql(128,off0) | qh(64,off128) | scales(16,off192) | d(f16,off208)。
#   每块 2 个 128 组, 组间指针推进 ql+=64, qh+=32, sc+=8。
#   y[l]=d·sc[is]·q1, y[l+32]=d·sc[is+2]·q2, y[l+64]=d·sc[is+4]·q3, y[l+96]=d·sc[is+6]·q4。
# Q8_0 (对照 ggml-quants.c:553-567, 34 字节/块): d(f16,2) + qs[32](i8); y = d·qs。
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8
GGML_Q6_K = 14
GGML_BF16 = 30
QK_K = 256
Q8_0_BLOCK = 34
Q6_K_BLOCK = 210


def dequant_q8_0(raw: torch.Tensor, n: int) -> torch.Tensor:
    """Q8_0 raw (uint8, 长度 (n//32)*34) → f32 [n]。对照 ggml-quants.c:553-567。"""
    nb = n // 32
    buf = raw.view(nb, Q8_0_BLOCK)
    d = buf[:, 0:2].view(torch.float16).float().view(-1)  # [nb]
    qs = buf[:, 2:34].view(torch.int8).float()           # [nb, 32]
    return (d.unsqueeze(1) * qs).reshape(-1)[:n]


def dequant_q6_K(raw: torch.Tensor, n: int) -> torch.Tensor:
    """Q6_K raw (uint8, 长度 (n//256)*210) → f32 [n]。
    逐字对照 ggml-quants.c dequantize_row_q6_K (本 checkout 无 dmin)。"""
    nb = n // QK_K
    buf = raw.view(nb, Q6_K_BLOCK)
    # 块布局 (ggml-common.h block_q6_K): ql[128](0) | qh[64](128) | scales[16](192) | d(f16)(208)
    d = buf[:, 208:210].view(torch.float16).float().view(-1)  # [nb]
    ql = buf[:, 0:128].view(torch.uint8).long()               # [nb,128]
    qh = buf[:, 128:192].view(torch.uint8).long()             # [nb,64]
    sc = buf[:, 192:208].view(torch.int8).float()             # [nb,16]
    l = torch.arange(32, device=raw.device)              # [32]
    is_ = (l // 16).long()                               # 0 或 1
    out = torch.empty(nb, QK_K, device=raw.device, dtype=torch.float32)
    for nn in range(2):
        qo, ho, so = 64 * nn, 32 * nn, 8 * nn            # 组间指针推进
        q1 = (ql[:, qo:qo + 32] & 0xF) | (((qh[:, ho:ho + 32] >> 0) & 3) << 4) - 32
        q2 = (ql[:, qo + 32:qo + 64] & 0xF) | (((qh[:, ho:ho + 32] >> 2) & 3) << 4) - 32
        q3 = (ql[:, qo:qo + 32] >> 4) | (((qh[:, ho:ho + 32] >> 4) & 3) << 4) - 32
        q4 = (ql[:, qo + 32:qo + 64] >> 4) | (((qh[:, ho:ho + 32] >> 6) & 3) << 4) - 32
        dd = d.unsqueeze(1)
        s = sc[:, so:so + 8]
        out[:, nn * 128 + l + 0] = dd * s[:, is_] * q1
        out[:, nn * 128 + l + 32] = dd * s[:, is_ + 2] * q2
        out[:, nn * 128 + l + 64] = dd * s[:, is_ + 4] * q3
        out[:, nn * 128 + l + 96] = dd * s[:, is_ + 6] * q4
    return out.reshape(-1)[:n]


def dequant(raw: torch.Tensor, n: int, ggml_type: int) -> torch.Tensor:
    """GGUF raw 字节 → f32 [n]。F32/F16/BF16 直接 cast; Q8_0/Q6_K 反量化。"""
    if ggml_type == GGML_F32:
        return raw[:n * 4].view(torch.float32)
    if ggml_type == GGML_F16:
        return raw[:n * 2].view(torch.float16).float()
    if ggml_type == GGML_BF16:
        u16 = raw[:n * 2].view(torch.uint16)
        # bf16: 高 16 位即 f32 的高位 → 左移 16 位解释为 f32
        return (u16.to(torch.int32) << 16).view(torch.float32)
    if ggml_type == GGML_Q8_0:
        return dequant_q8_0(raw, n)
    if ggml_type == GGML_Q6_K:
        return dequant_q6_K(raw, n)
    raise ValueError(f"unsupported quant type {ggml_type}")


class QuantLinear(nn.Module):
    """量化驻留线性层。quantized 路径: raw uint8 驻留, forward 现场反量化 (不保留 bf16 权重)。
    float 路径: weight Parameter 直接持有 (CPU 冒烟)。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # R3: meta 占位 (零 host / 零 VRAM / 零触碰) — 对照 vllm meta 哑模型。
        # 无 kaiming_uniform_ 全触碰 (Round-2 构造期 OOM 死因)。
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device="meta"))
        self.bias = nn.Parameter(torch.empty(out_features, device="meta")) if bias else None
        self._raw = None
        self._ggml_type = None
        self._shape = None

    def load_quantized(self, raw: torch.Tensor, shape: tuple, ggml_type: int):
        """shape = torch 序 (out, in)。raw 原样驻留 (loader 已放目标设备), 释放 meta 占位。"""
        self._raw = raw
        self._ggml_type = ggml_type
        self._shape = shape
        self.weight = None

    def load_dequantized(self, w: torch.Tensor):
        """float 路径 (CPU/bf16 合成冒烟)。meta 占位先物化再 copy_ — 无全触碰 init。"""
        if self.weight.is_meta:
            self.weight = nn.Parameter(
                torch.empty(self.weight.shape, dtype=w.dtype, device=w.device),
                requires_grad=False)
        with torch.no_grad():
            self.weight.copy_(w.to(self.weight.dtype))

    def _weight(self, dtype):
        if self._raw is not None:
            n = self._shape[0] * self._shape[1]
            w = dequant(self._raw, n, self._ggml_type).to(dtype)
            return w.view(self._shape)
        # 无 loader 路径 (k6 合成前向): meta 占位惰性物化到 CPU — torch.empty 不触碰,
        # 无 kaiming/normal_/zeros_/ones_。dtype 沿用模块当前 dtype (k6 .to(bf16) 后为 bf16)。
        if self.weight.is_meta:
            self.weight = nn.Parameter(
                torch.empty(self.weight.shape, dtype=self.weight.dtype, device="cpu"),
                requires_grad=False)
        return self.weight

    def forward(self, x):
        w = self._weight(x.dtype)
        return F.linear(x, w, self.bias)


class QuantEmbedding(nn.Module):
    """token_embd 量化驻留 (Q8_0 [vocab, hidden])。forward 按行 gather 后反量化 (行 gather dequant)。"""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # R3: meta 占位 (零 host / 零 VRAM / 零触碰) — 无 normal_ 全触碰
        # (Round-2 构造期 OOM 死因之一: token_embd f32 5.09GB × 触碰)。
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device="meta"))
        self._raw = None
        self._ggml_type = None
        self._shape = None

    def load_quantized(self, raw: torch.Tensor, shape: tuple, ggml_type: int):
        """raw 原样驻留 (loader 已放目标设备), 释放 meta 占位。"""
        self._raw = raw
        self._ggml_type = ggml_type
        self._shape = shape
        self.weight = None

    def load_dequantized(self, w: torch.Tensor):
        """float 路径 (CPU/bf16 合成冒烟)。meta 占位先物化再 copy_ — 无全触碰 init。"""
        if self.weight.is_meta:
            self.weight = nn.Parameter(
                torch.empty(self.weight.shape, dtype=w.dtype, device=w.device),
                requires_grad=False)
        with torch.no_grad():
            self.weight.copy_(w.to(self.weight.dtype))

    def forward(self, input_ids):
        if self._raw is None:
            # 无 loader 路径 (k6 合成前向): meta 占位惰性物化到 CPU — torch.empty 不触碰。
            if self.weight.is_meta:
                self.weight = nn.Parameter(
                    torch.empty(self.weight.shape, dtype=self.weight.dtype,
                                device="cpu"),
                    requires_grad=False)
            return F.embedding(input_ids, self.weight)
        # 行 gather 反量化: 每行 embedding_dim 个元素 (32 的倍数) → 只反量化用到的行
        rows_per_block = self.embedding_dim // 32        # Q8_0: 每行 160 块
        row_bytes = rows_per_block * Q8_0_BLOCK
        raw_rows = self._raw.view(-1, row_bytes)         # [vocab, row_bytes]
        gathered = raw_rows[input_ids.reshape(-1)]       # [T, row_bytes]
        w = dequant(gathered, gathered.shape[0] * self.embedding_dim, self._ggml_type) \
            .view(-1, self.embedding_dim)
        return w.view(*input_ids.shape, self.embedding_dim)


def is_quantized_module(m) -> bool:
    return isinstance(m, (QuantLinear, QuantEmbedding))
