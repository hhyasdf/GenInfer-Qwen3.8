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
#   - QuantLinear: forward 时 dequant(raw)→f32 做 F.linear (R11: 对齐 llama.cpp F32 计算路径),
#     不保留完整 f32 权重 (GEMM 后释放)。
#     峰值 = 1 个 GEMM 的反量化权重 (FFN 17408×5120×4B ≈ 356MB), 对照 D-6a VRAM 预算。
#   - QuantEmbedding: token_embd Q8_0 [248320,5120] 按行 gather 后反量化 (T×5120×2B),
#     避免 2.5GB 全量反量化。
#   - 同时保留 float 路径 (load_dequantized) 供 CPU/bf16 合成前向冒烟。
#
# Round-4 低温 dequant (修 R3 L3 死因 — prefill dequant f32+int64 临时 ~4.5GB → GPU0 16GB 触顶):
#   dequant_q6_K / dequant_q8_0 接受 dtype 参数; R11 起 GEMM 路径 (QuantLinear._weight) 传
#   torch.float32 (对齐 llama.cpp LLM_DEFAULT_FTYPE=F32 计算路径) — 逐块元素级位提取直接写 f32:
#   ql/qh/sc 是零拷贝 uint8/int8 view 做位提取, q 恰好落 int8 (-32..31),
#   只在 ≤65536 块的 chunk 内物化 f32 乘积 (out_dtype=f32 时乘积直落, 无二次 cast)。
#   整层临时仅 f32 GEMM 权重本身 (FFN 17408×5120×4B ≈ 356MB, 每-GEMM 预算, GEMM 后释放)。
#   语义与 llama.cpp dequantize_row_q6_K (ggml-quants.c:1939-1968) /
#   dequantize_row_q8_0 (ggml-quants.c:553-567) / 寄存器级 dequantize_q6_K
#   (ggml-cuda/dequantize.cuh:246-266, y = d·sc[is]·(q−32)) 位级一致;
#   torch 引擎不调用 llama.cpp CUDA kernel — 纯 torch 重实现 (sm89 dequant+cuBLAS 路径语义)。
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


def dequant_q8_0(raw: torch.Tensor, n: int, dtype=None) -> torch.Tensor:
    """Q8_0 raw (uint8, 长度 (n//32)*34) → [n]。dtype=None → f32 (float 路径)。
    y[i*32+j] = qs[j]·d (ggml-quants.c:553-567), 逐块元素级: qs 是零拷贝 int8 view,
    只在 chunk (≤65536 块) 内物化 f32 乘积再 cast 到 dtype。
    R4 低温路径: 不再整层 f32 物化 (旧实现 f32 [n] 全驻留, LMHead 5.09GB → 双 cast 到 bf16)。"""
    nb = n // 32
    buf = raw.view(nb, Q8_0_BLOCK)
    out_dtype = torch.float32 if dtype is None else dtype
    d = buf[:, 0:2].view(torch.float16).float().reshape(-1)  # [nb] f32 (d, f16→f32 无损)
    qs = buf[:, 2:34].view(torch.int8)            # [nb,32] int8 (零拷贝 view, 无物化)
    out = torch.empty(nb, 32, device=raw.device, dtype=out_dtype)
    CH = 65536
    for c0 in range(0, nb, CH):
        c1 = min(c0 + CH, nb)
        out[c0:c1] = (d[c0:c1].unsqueeze(1) * qs[c0:c1].float()).to(out_dtype)
    return out.reshape(-1)[:n]


def dequant_q6_K(raw: torch.Tensor, n: int, dtype=None) -> torch.Tensor:
    """Q6_K raw (uint8, 长度 (n//256)*210) → [n]。dtype=None → f32 (float 路径 / 行 gather)。
    逐块元素级反量化, 与旧实现和 C 参考位级一致 (y = d·sc[is]·(q−32), q 为 6 位
    (ql 低 4 位 | qh 2 位<<4)): ggml-quants.c:1939-1968 / ggml-cuda/dequantize.cuh:246-266。
    R4 低温路径 (修 Round-3 L3 死因): 不整层 f32 out, 不整层 int64 ql/qh —
      ql/qh/sc 是零拷贝 uint8/int8 view 做位提取, q 恰好落 int8 (-32..31),
      只在 chunk (≤65536 块) 内物化 f32 乘积, 直接 cast 到 dtype。
      out 张量本身 (FFN 17408×5120 bf16 ≈ 170MB) 是唯一整层临时 = 每-GEMM 预算,
      GEMM 后释放; 每 chunk 片临时 ≤ ~20MB。
      旧实现整层 f32 356MB + int64 ql/qh 890MB + int64 q1..q4 356MB → stage0 ~4.5GB → OOM。
    语义对照 dequantize.cuh:246-266 (寄存器级, sm89 dequant+cuBLAS 路径等价语义):
      y[0]=d·sc[is]·((ql&0xF)|((qh>>0)&3)<<4 −32), y[32]=…sc[is+2]·((ql+32&0xF)|((qh>>2)&3)<<4 −32),
      y[64]=…sc[is+4]·((ql>>4)|((qh>>4)&3)<<4 −32), y[96]=…sc[is+6]·((ql+32>>4)|((qh>>6)&3)<<4 −32)。
    torch 引擎不调用 llama.cpp CUDA kernel — 纯 torch 重实现其语义。"""
    nb = n // QK_K
    buf = raw.view(nb, Q6_K_BLOCK)
    out_dtype = torch.float32 if dtype is None else dtype
    d = buf[:, 208:210].view(torch.float16).float().reshape(-1)  # [nb] f32 (d)
    sc = buf[:, 192:208].view(torch.int8)                  # [nb,16] int8 (零拷贝 view)
    out = torch.empty(nb, QK_K, device=raw.device, dtype=out_dtype)
    l = torch.arange(32, device=raw.device)
    is_ = l // 16                                          # scale 子索引 0/1 (0..15→0, 16..31→1)
    CH = 65536
    for c0 in range(0, nb, CH):
        c1 = min(c0 + CH, nb)
        dc = d[c0:c1].unsqueeze(1)                         # [c,1] f32
        for nn in range(2):
            qo, ho, so = 64 * nn, 32 * nn, 8 * nn          # 组间指针推进
            lo_a = buf[c0:c1, qo:qo + 32]                  # uint8 view (q1 低 / q3 高 nibble, ql 基址 @0)
            lo_b = buf[c0:c1, qo + 32:qo + 64]             # uint8 view (q2 / q4, ql 基址 @0)
            qh = buf[c0:c1, 128 + ho:128 + ho + 32]        # uint8 view (qh 基址 @128, 组偏移 ho=32*nn)
            s = sc[c0:c1, so:so + 8].float()               # [c,8] f32
            for k in range(4):
                src = lo_a if k % 2 == 0 else lo_b
                qlo = (src & 0xF) if k < 2 else (src >> 4)  # uint8 [c,32]
                qhi = (qh >> (2 * k)) & 3                   # uint8 [c,32] (qh>>0 / >>2 / >>4 / >>6)
                q = (qlo | (qhi << 4)).to(torch.int8) - 32  # int8 [c,32], -32..31 (精确)
                s_exp = s[:, is_ + 2 * k]                   # f32 [c,32] (sc[is], sc[is+2], sc[is+4], sc[is+6])
                out[c0:c1, nn * 128 + 32 * k: nn * 128 + 32 * (k + 1)] = \
                    (dc * s_exp * q.float()).to(out_dtype)
    return out.reshape(-1)[:n]


def dequant(raw: torch.Tensor, n: int, ggml_type: int, dtype=None) -> torch.Tensor:
    """GGUF raw 字节 → [n]。dtype=None → f32 (float 路径, 行为与 R3 一致);
    dtype=bf16 等 → Q8_0/Q6_K 直接反量化进该 dtype (R4 低温: GEMM 路径传 x.dtype,
    免 f32→bf16 双物化); F32/F16/BF16 零拷贝 view + 可选 cast。"""
    if ggml_type == GGML_F32:
        w = raw[:n * 4].view(torch.float32)
        return w if dtype is None else w.to(dtype)
    if ggml_type == GGML_F16:
        w = raw[:n * 2].view(torch.float16).float()
        return w if dtype is None else w.to(dtype)
    if ggml_type == GGML_BF16:
        u16 = raw[:n * 2].view(torch.uint16)
        # bf16: 高 16 位即 f32 的高位 → 左移 16 位解释为 f32
        w = (u16.to(torch.int32) << 16).view(torch.float32)
        return w if dtype is None else w.to(dtype)
    if ggml_type == GGML_Q8_0:
        return dequant_q8_0(raw, n, dtype)
    if ggml_type == GGML_Q6_K:
        return dequant_q6_K(raw, n, dtype)
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
            # R4 低温路径 + R11: 直接反量化到 GEMM dtype — 不整层物化后双 cast。
            # R11 起 dtype 恒为 torch.float32 (对齐 llama.cpp LLM_DEFAULT_FTYPE=F32 计算路径;
            # dequantize.cuh:246-263 的 d·sc·(q−32) f32 乘积直落 dst_t=float)。
            # dequant 临时 = 这块 f32 权重本身 (每-GEMM 预算: FFN ≤356MB / LMHead 4.74GiB),
            # F.linear 后立即释放。dequant(..., dtype=f32) 为 R4 低温路径原生支持
            # (out_dtype = torch.float32, chunk 内 f32 乘积直落, 无二次 cast)。
            w = dequant(self._raw, n, self._ggml_type, dtype)
            # ⚠️ R12 防误改: self._shape 是 GGUF dims 反序 (torch 序 [out,in])。
            # 文件字节是 ne[0] 最内层的 row-major (gguf.cpp nb[0]=type_size, nb[1]=ne[0]*nb[0];
            # llama.cpp 加载期无转置) — 故 dequant(raw).view(反序) 数学上 == view(GGUF序).T,
            # 已是真转置 (attn_qkv vs llama.cpp matmul 0.000e+00; ssm_conv1d 5.96e-08)。
            # 切勿再 .T 或改 view 顺序 — 实测直接 view(GGUF序).T 会乱权 (Δ up to 7.0)。
            return w.view(self._shape)
        # 无 loader 路径 (k6 合成前向): meta 占位惰性物化到 CPU — torch.empty 不触碰,
        # 无 kaiming/normal_/zeros_/ones_。dtype = 请求的 GEMM dtype (R11: f32)。
        if self.weight.is_meta:
            self.weight = nn.Parameter(
                torch.empty(self.weight.shape, dtype=dtype, device="cpu"),
                requires_grad=False)
        return self.weight

    def forward(self, x):
        # R11: 全部 GEMM fp32 计算 (prefill + decode) — 反量化直落 f32, F.linear 在 f32 上
        # 执行 (cuBLAS f32 累加 → f32 输出), 对齐 llama.cpp 的 LLM_DEFAULT_FTYPE=F32 计算路径
        # (GGUF general.file_type=18)。x 若非 f32 则上转换 (生产流自 R11 起恒为 f32, 此处为防护)。
        w = self._weight(torch.float32)
        return F.linear(x.to(torch.float32), w, self.bias)


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
