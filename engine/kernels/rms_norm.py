# RMSNorm — Qwen3.5 hybrid 的三种归一化变体 (02_qwen3_5_hybrid.md)。
#
# 三种变体 (eps 统一 1e-6):
#   1. RMSNorm          — 标准归一化, weight 初始化为 1。
#   2. RMSNorm(init_zeros=True) — 零初始化归一化 (weight 初始 0, 前向初期等价恒等残差通路)。
#   3. GatedRMSNorm     — 带门控的归一化, 输出 = norm(x) * silu(gate)。
#      用于 GDN 线性注意力的输出归一化 (attn_output_gate=True 语义)。
#
# 契约: forward 不改变输入 shape; 返回 (normalized, ...) 视调用方需要。
# 实现: 优先 vllm._custom_ops.rms_norm (bf16 in-place); 不可用时 torch 参考实现。
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # vLLM C++ kernel 路径 (ENV_CAPABILITY: vllm._custom_ops AVAILABLE)
    from vllm import _custom_ops as _vops
    _HAS_VLLM_OPS = True
except Exception:  # pragma: no cover - fallback 路径
    _vops = None
    _HAS_VLLM_OPS = False


class RMSNorm(nn.Module):
    """RMSNorm。两种初始化模式:
      - 标准 (ones_plus=False): out = x * rsqrt(mean(x²)+eps) * weight, weight init 1。
      - Qwen3.5 zeros-init (ones_plus=True): out = x * rsqrt(mean(x²)+eps) * (1 + weight),
        weight init 0 (LOAD-Q35-004: 公式 (1+w) 在 forward 处理, 加载直接 copy)。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 init_zeros: bool = False, ones_plus: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.ones_plus = ones_plus
        if ones_plus:
            self.weight = nn.Parameter(torch.zeros(hidden_size))
        elif init_zeros:
            self.weight = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight = nn.Parameter(torch.ones(hidden_size))

    def _scale(self, w: torch.Tensor) -> torch.Tensor:
        return 1.0 + w if self.ones_plus else w

    def _can_fused(self, x: torch.Tensor) -> bool:
        # vllm rms_norm 仅 CUDA 后端; bf16 + contiguous + cuda + 非 ones_plus。
        # Issue 9 (R5 死点): weight dtype 必须 == 输入 dtype。GGUF 的 F32 norm 权重被
        # weight_loader._bind 绑为 f32 Parameter (weight_loader.py:230 的
        # obj.copy_(w.to(obj.dtype)) 对 F32 是 no-op — obj.dtype 本就是 f32), 而生产流
        # 是 bf16 (qwen.py:403 token_embd 输出显式 cast bf16)。vllm fused 内核按**输入**
        # 类型派发并读 weight.data_ptr<scalar_t>() (layernorm_kernels.cu:211/:222, fused
        # 变体 :243 显式 TORCH_CHECK weight.scalar_type()==input.scalar_type()) → f32
        # weight + bf16 input = "expected scalar type BFloat16 but found Float" 硬崩。
        # 失配时 fallback 到下方 dtype 无关的 torch 参考路径 (f32 累加 — 与 llama.cpp
        # 原生 F32 RMSNorm 语义一致: ggml-cpu/ops.cpp:3989-4004 switch 只处理 F32 其余
        # ABORT, fused MUL 读 float* weight; ggml-cuda/norm.cu:478-500 断言 F32)。
        # 对照 vllm 不变量 (layernorm.py:104-116 weight 按模型 dtype 创建 → weight.dtype
        # == 激活 dtype 结构性保证), 本引擎量化驻留路径无模型级 cast, 故由内核门禁守卫。
        return (_HAS_VLLM_OPS and x.dtype == torch.bfloat16
                and x.is_contiguous() and x.device.type == "cuda"
                and self.weight.dtype == x.dtype)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        # 首层: residual=None → 仅归一化, 返回 (normalized, x.clone())
        # 后续层: residual 非 None → fused_add_rms_norm 语义 (residual+=x; x=rms(residual))
        w = self._scale(self.weight)
        if residual is None:
            if (not self.ones_plus) and self._can_fused(x):
                out = torch.empty_like(x)
                _vops.rms_norm(out, x, self.weight, self.eps)
                return out, x.clone()
            rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
            out = (x.float() * rms * w.float()).to(x.dtype)
            return out, x.clone()
        res = x + residual
        if (not self.ones_plus) and self._can_fused(res):
            out = torch.empty_like(res)
            _vops.rms_norm(out, res, self.weight, self.eps)
            return out, res
        rms = torch.rsqrt(res.float().pow(2).mean(-1, keepdim=True) + self.eps)
        out = (res.float() * rms * w.float()).to(res.dtype)
        return out, res


class GatedRMSNorm(RMSNorm):
    """带门控的 RMSNorm: out = norm(x) * silu(gate)。

    gate 是调用方提供的与 x 同 shape 的张量 (GDN 输出门控 / attn_output_gate)。
    weight 语义与标准 RMSNorm 相同。
    """

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None,
                residual: torch.Tensor | None = None):
        normalized, res = super().forward(x, residual)
        if gate is not None:
            normalized = normalized * F.silu(gate)
        return normalized, res
