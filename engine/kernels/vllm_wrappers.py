# Phase 1 — 7 个 vLLM 黑盒 kernel wrapper (kernel_contracts.md §九)。
#
# 环境 (phase_report/ENV_CAPABILITY.md):
#   - vllm._custom_ops: AVAILABLE  → rms_norm / silu_and_mul 走 C++ kernel
#   - flash_attn:        UNAVAILABLE → attention 走 F.scaled_dot_product_attention
#
# 每个 wrapper: 优先 vLLM C++ 路径, 不可用时 torch 参考实现 (数值等价)。
import torch
import torch.nn.functional as F

try:  # KERNEL-008: import vllm._C 触发 torch.ops._C.silu_and_mul 注册
    from vllm import _custom_ops as ops  # noqa: F401
    import vllm._C  # noqa: F401
    _HAS_VLLM = True
except Exception:  # pragma: no cover
    ops = None
    _HAS_VLLM = False

try:
    from flash_attn import flash_attn_varlen_func  # noqa: F401
    _HAS_FA_VARLEN = True
except Exception:
    _HAS_FA_VARLEN = False

try:
    from flash_attn.flash_attn_interface import flash_attn_with_kvcache  # noqa: F401
    _HAS_FA_KVCACHE = True
except Exception:
    _HAS_FA_KVCACHE = False


# ---------------------------------------------------------------------------
# KERNEL-001 / KERNEL-002: rms_norm / fused_add_rms_norm
# ---------------------------------------------------------------------------
def _can_vllm(x: torch.Tensor) -> bool:
    """vllm C++ op 仅 CUDA 后端: bf16 + contiguous + cuda。"""
    return (_HAS_VLLM and x.dtype == torch.bfloat16
            and x.is_contiguous() and x.device.type == "cuda")


def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
             epsilon: float) -> torch.Tensor:
    """rms_norm(out!, input, weight, eps)。bf16, contiguous。in-place 写 out。"""
    if _can_vllm(input):
        ops.rms_norm(out, input, weight, epsilon)
        return out
    rms = torch.rsqrt(input.float().pow(2).mean(-1, keepdim=True) + epsilon)
    out.copy_((input.float() * rms * weight.float()).to(input.dtype))
    return out


def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, epsilon: float):
    """fused_add_rms_norm(input!, residual!, weight, eps)。双 in-place:
    residual += input; input = rms_norm(residual)。返回 (input, residual)。"""
    if _can_vllm(input):
        ops.fused_add_rms_norm(input, residual, weight, epsilon)
        return input, residual
    residual += input
    rms = torch.rsqrt(residual.float().pow(2).mean(-1, keepdim=True) + epsilon)
    input.copy_((residual.float() * rms * weight.float()).to(residual.dtype))
    return input, residual


# ---------------------------------------------------------------------------
# KERNEL-003: silu_and_mul
# ---------------------------------------------------------------------------
def silu_and_mul(out: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    """silu_and_mul(out!, input)。input[..., :d]=gate, input[..., d:]=up;
    out = silu(gate) * up。d = input.shape[-1] // 2。"""
    d = input.shape[-1] // 2
    gate, up = input[..., :d], input[..., d:]
    if (_HAS_VLLM and input.dtype in (torch.bfloat16, torch.float16)
            and input.is_contiguous() and input.device.type == "cuda"):
        ops.silu_and_mul(out, input)
        return out
    out.copy_(F.silu(gate.float()) * up.float())
    return out


# ---------------------------------------------------------------------------
# KERNEL-004 / KERNEL-005: RoPE (cos/sin cache + apply)
# ---------------------------------------------------------------------------
def _rotary_embedding_cache(max_seq_len: int, head_dim: int, base: float,
                            device, dtype=torch.float32):
    """KERNEL-005: cos/sin cache 工厂。返回 (cos, sin), 各 [max_seq_len, head_dim/2]。
    Qwen3.5 partial rotary: 只前 head_dim*0.25 维旋转, 其余 passthrough (在 apply 处理)。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype).float() / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor,
                     cos: torch.Tensor, sin: torch.Tensor):
    """KERNEL-004: 把 RoPE 应用到 q/k。q:[*, n_heads, head_dim], k:[*, n_kv, head_dim]。
    cos/sin:[seq, head_dim/2]。只旋转前 rope_dim 维 (partial rotary), 其余 passthrough。"""
    def _apply(x, c, s):
        # x: [..., n, d]; c, s: [seq, d/2] → 广播到 [..., n, d/2]
        d_rot = c.shape[-1] * 2
        x_rot, x_pass = x[..., :d_rot], x[..., d_rot:]
        x1, x2 = x_rot[..., 0::2], x_rot[..., 1::2]
        c_b = c.unsqueeze(0).expand_as(x1)
        s_b = s.unsqueeze(0).expand_as(x1)
        out = torch.empty_like(x_rot)
        out[..., 0::2] = x1 * c_b - x2 * s_b
        out[..., 1::2] = x2 * c_b + x1 * s_b
        return torch.cat([out, x_pass], dim=-1) if x_pass.numel() else out
    return _apply(q, cos, sin), _apply(k, cos, sin)


# ---------------------------------------------------------------------------
# KERNEL-006 / KERNEL-007: attention (flash_attn 不可用 → SDPA)
# ---------------------------------------------------------------------------
def attention_fwd(q, k, v, is_prefill: bool, past_len: int = 0):
    """统一 attention 前向。q:[B,nh,T,d], k,v:[B,nkv,T,d] (或 paged cache)。
    flash_attn UNAVAILABLE → F.scaled_dot_product_attention (ENV_CAPABILITY: SAFE)。
    prefill: causal; decode: 无 causal (已有完整 KV history)。"""
    if _HAS_FA_VARLEN and is_prefill:
        # ragged 3D [tokens, heads, dim] 路径 (物理 trace prefill)
        out = flash_attn_varlen_func(q, k, v, None, None,
                                     q.shape[2], q.shape[2], causal=True)
        return out
    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_prefill)
    return out


def attention_decode(q, key_cache, value_cache, kv_len, block_table,
                     scale=None):
    """decode attention。flash_attn_with_kvcache UNAVAILABLE → SDPA 展开 paged KV。
    q:[B,nh,1,d]; cache:[num_blocks, block_size, nkv, d]。"""
    if _HAS_FA_KVCACHE:
        return flash_attn_with_kvcache(
            q.unsqueeze(2), key_cache, value_cache, kv_len, block_table,
            scale=scale, causal=False).squeeze(2)
    # SDPA fallback: 按 block_table 索引展开 paged KV → contiguous [B, nkv, T, d]
    bsz = q.shape[0]
    nkv = key_cache.shape[2]
    d = q.shape[-1]
    seq = kv_len.item() if torch.is_tensor(kv_len) else int(kv_len)
    k_out = key_cache.new_zeros(bsz, nkv, seq, d)
    v_out = value_cache.new_zeros(bsz, nkv, seq, d)
    for i in range(bsz):
        for pos in range(seq):
            blk = block_table[i, pos // 256].item()
            off = pos % 256
            k_out[i, :, pos] = key_cache[blk, off]
            v_out[i, :, pos] = value_cache[blk, off]
    # GQA: 复制 KV 到 q heads
    rep = q.shape[1] // nkv
    k_out = k_out.repeat_interleave(rep, dim=1)
    v_out = v_out.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q, k_out, v_out, is_causal=False)
    return out


def kernel_wrappers_available() -> dict:
    """KERNEL-SH-001 依赖探测 (供 ENV_CAPABILITY 对照)。"""
    return {
        "vllm_custom_ops": _HAS_VLLM,
        "flash_attn_varlen": _HAS_FA_VARLEN,
        "flash_attn_with_kvcache": _HAS_FA_KVCACHE,
        "sdpa_safe": True,  # ENV_CAPABILITY: F.scaled_dot_product_attention SAFE
    }
