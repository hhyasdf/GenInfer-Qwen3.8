# Phase 5(PP variant) + Phase 6 — Qwen3.5 hybrid model。
#
# 架构: 64 主层 (48 GDN 线性注意力 + 16 FullAttention, FA 在 il%4==3) + 1 MTP blk.64。
# norm 归属 (FM-003): decoder 层持有 input_layernorm + post_attention_layernorm,
#   fused_add_rms_norm 必用本层 weight; attention/GDN 层只做投影/recurrence/attention, 不再自归一。
# PP 变体: 每 stage 只建本区间层; MLP 用纯 nn.Linear (无 ColumnParallel/RowParallel — TP only, 禁)。
#   - FA 层: per-stage KV (按层切, 不按 head 切)。
#   - GDN 层: per-layer recurrent state [n_v, Dk, Dv] fp32, 归本 stage。
# 首 stage 持 token_embd, 末 stage 持 LMHead; 中间 stage 传 IntermediateTensors。
# R5 (Issue 8): 模型有 MTP 块时, 边界增传 token_embd 键 (首 rank 输入 token 的
# embedding, llama.cpp graph_mtp 双图输入 inp->embd 语义), 末 rank MTP 块从边界读。
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..kernels import RMSNorm
from ..pp import get_pp_group, IntermediateTensors, make_empty_intermediate_tensors, PPError
from ..weights.quant_linear import QuantLinear, QuantEmbedding


def _logits_finiteness_diag(logits):
    """② logits 有限性诊断 (引擎侧, 供 Verification L3 采集)。
    只读: torch.isfinite 全量扫描 + 首个非有限 logit 的位置/数值/占比。
    不改任何数值路径, 不影响 greedy 采样 (下游 argmax 消费的是原 logits)。
    单行 [LOGITS-DIAG] 输出, 可 grep。"""
    fin = torch.isfinite(logits)
    total = int(fin.numel())
    n_fin = int(fin.sum().item())
    if n_fin == total:
        print(f"[LOGITS-DIAG] shape={tuple(logits.shape)} dtype={logits.dtype} "
              f"finite={total}/{total} (all finite)", flush=True)
        return
    first = (~fin).nonzero()[0].tolist()
    val = float(logits[tuple(first)])
    print(f"[LOGITS-DIAG] shape={tuple(logits.shape)} dtype={logits.dtype} "
          f"finite={n_fin}/{total} non_finite_ratio={(total - n_fin) / total:.6f} "
          f"first_nonfinite pos={first} value={val}", flush=True)


# ---------------------------------------------------------------------------
# GDN (Gated Delta Net) 线性注意力 (recurrent, state [n_v, Dk, Dv] fp32 常数)
# ---------------------------------------------------------------------------
class GatedDeltaNet(nn.Module):
    """GDN 线性注意力。输入 = 已归一化 hidden [B,T,hidden]。
    投影 (对照 llama.cpp qwen35.cpp:340-470, 量化驻留 QuantLinear):
      attn_qkv    [hidden, key_dim*2 + value_dim] → q[n_k,dk] k[n_k,dk] v[n_v,dv]
      attn_gate   [hidden, value_dim]             → z gate [n_v,dv]
      ssm_alpha   [hidden, n_v]  → decay 基
      ssm_beta    [hidden, n_v]  → update gate (sigmoid)
      ssm_out     [value_dim, hidden] → output 投影
      ssm_a       [n_v] (F32, 直接驻留) = -A_log.exp() (负)
      ssm_dt.bias [n_v] (F32)  → 加到 alpha
      ssm_conv1d  [qkv_dim, K] (F32) depthwise causal conv + silu
      ssm_norm    [dv] (F32)  → 输出门控 norm (per-head, dv 空间)
    GQA: value head v → key head v%n_k (tile 取模, repeat(1,1,ratio), 对齐 llama.cpp ggml_repeat_4d)。
    Compute 修复 (Round-2):
      (1) conv1d depthwise + silu 作用于 qkv_mixed (R10: 持久状态环 — 前置 cat([state, x])
          全段卷积, 输出取后 T 行, 回写卷积前尾部 K-1=3 原始行;
          对照 llama.cpp delta-net-base.cpp:449-525 build_conv_state)
      (2) q/k L2 归一 (eps=1e-6); q 乘 1/sqrt(dk)
      (3) gate = softplus(alpha + ssm_dt) * ssm_a (负) → decay = exp(gate)
      (4) beta = sigmoid(ssm_beta @ x)
    Recurrence (gated delta rule):
      S_t = S_{t-1} * exp(gate_t) + k_t ⊗ (beta_t * (v_t - S_{t-1} k_t))
      o_t = S_t q_t
    输出: gated = RMSNorm_dv(o) * silu(z); out = ssm_out @ gated。
    """

    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.hidden_size
        self.n_k = cfg.linear_num_key_heads      # 16
        self.n_v = cfg.linear_num_value_heads    # 48
        self.dk = cfg.linear_key_head_dim        # 128
        self.dv = cfg.linear_value_head_dim      # 128
        self.key_dim = self.n_k * self.dk        # 2048
        self.value_dim = self.n_v * self.dv      # 6144
        self.qkv_dim = self.key_dim * 2 + self.value_dim  # 10240
        self.gqa_ratio = self.n_v // self.n_k    # 3
        self.conv_kernel = cfg.linear_conv_kernel_dim     # 4
        self.wqkv = QuantLinear(hidden, self.qkv_dim)
        self.wqkv_gate = QuantLinear(hidden, self.value_dim)
        self.ssm_alpha = QuantLinear(hidden, self.n_v)
        self.ssm_beta = QuantLinear(hidden, self.n_v)
        self.ssm_out = QuantLinear(self.value_dim, hidden)
        # 小 F32 张量 (非量化, 直接驻留 nn.Parameter)
        self.ssm_a = nn.Parameter(torch.empty(self.n_v))            # 负
        self.ssm_dt = nn.Parameter(torch.empty(self.n_v))           # dt bias
        self.ssm_conv1d = nn.Parameter(torch.empty(self.qkv_dim, self.conv_kernel))
        self.ssm_norm = nn.Parameter(torch.empty(self.dv))
        self.rms_eps = cfg.rms_norm_eps

    def _conv1d(self, qkv_mixed, conv_state):
        """depthwise causal conv (kernel=conv_kernel) + silu — 持久状态环语义。
        qkv_mixed:[B,T,C] (卷积前原始输入), conv_state:[B,K-1,C] (持久环, 与 qkv_mixed 同 dtype)。
        对照 llama.cpp: delta-net-base.cpp:466-472 (环 reshape [K-1,C] + concat 前置
        build conv_input) + qwen35.cpp:391 (ggml_ssm_conv 全段卷积, n_t = ne0-K+1 = 取后 T 行)
        + delta-net-base.cpp:479-496 (状态更新 = conv_input 尾部 K-1 行, 卷积前原始行)。
        inp = cat([conv_state, qkv_mixed], dim=时间维) 全段 conv1d, 输出取后 T 行;
        conv_state ← inp 的尾部 K-1 行 (卷积前原始输入, 非卷积后) 供下一步。
        返回 (out [B,T,C], conv_state_new [B,K-1,C])。"""
        B, T, C = qkv_mixed.shape
        K = self.conv_kernel
        inp = torch.cat([conv_state, qkv_mixed], dim=1)   # [B, K-1+T, C] 环前置
        # R12 D2: tap 反转 — llama.cpp ssm-conv.cu 的卷积是 output[i] = Σ_j w[j]·x[i+j]
        # (w[0]→最旧位置, w[K-1]→当前位置); 本引擎求和式 out[t] = Σ_j inp[t-j]·w[j]
        # (w[0]→当前, w[K-1]→最旧), 方向相反 → 使用前对卷积核 flip(1) 对齐。
        w = self.ssm_conv1d.to(qkv_mixed.dtype).flip(1)   # [C, K]
        L = K - 1 + T
        out = inp.new_zeros(B, L, C)
        out += inp * w[:, 0]                              # j=0 全段 ([C] 沿 C 维广播)
        for j in range(1, K):
            out[:, j:] += inp[:, :L - j] * w[:, j]
        return F.silu(out[:, K - 1:]), inp[:, T:T + K - 1]  # 后 T 行 + 环回写(尾部 K-1 原始行)

    def _split_qkv(self, qkv_mixed, B, T):
        q = qkv_mixed[..., :self.key_dim].view(B, T, self.n_k, self.dk)
        k = qkv_mixed[..., self.key_dim:self.key_dim * 2].view(B, T, self.n_k, self.dk)
        v = qkv_mixed[..., self.key_dim * 2:].view(B, T, self.n_v, self.dv)
        return q, k, v

    def _recurrent(self, q, k, v, gate, beta, state, B, T):
        """q,k:[B,T,n_v,dk] (已 repeat(1,1,ratio) tile 取模) v:[B,T,n_v,dv] gate,beta:[B,T,n_v]。
        state:[B,n_v,dk,dv] fp32。返回 (output [B,T,n_v,dv], state)。"""
        o = torch.empty(B, T, self.n_v, self.dv, device=state.device, dtype=torch.float32)
        state = state.float()
        for t in range(T):
            qt = q[:, t].float()
            kt = k[:, t].float()
            vt = v[:, t].float()
            dec = torch.exp(gate[:, t].float())          # [B,n_v] decay (<1)
            bt = beta[:, t].float()                      # [B,n_v] update
            for j in range(self.n_v):
                sk = torch.einsum('bij,bi->bj', state[:, j], kt[:, j])     # S @ k
                new_v = bt[:, j, None] * (vt[:, j] - dec[:, j, None] * sk)  # delta = β(v − g·S^T k), 先衰减再读出 (llama.cpp KDA=false)
                state[:, j] = state[:, j] * dec[:, j, None] + kt[:, j].unsqueeze(-1) * new_v.unsqueeze(-2)
                o[:, t, j] = torch.einsum('bij,bi->bj', state[:, j], qt[:, j])
        return o, state

    def _gated_norm(self, o, z):
        """o,z:[B,T,n_v,dv] → RMSNorm_dv(o) * silu(z) (per-head, dv 空间)。"""
        B, T, nv, dv = o.shape
        orf = o.reshape(-1, dv).float()
        mean = (orf.pow(2).mean(-1, keepdim=True) + self.rms_eps).rsqrt()
        orf = (orf * mean).to(o.dtype) * self.ssm_norm.to(o.dtype)
        return (orf.reshape(B, T, nv, dv) * F.silu(z)).reshape(B, T, self.value_dim)

    def forward(self, x, gdn_state):
        """x:[B,T,hidden] (已归一化)。gdn_state: (ssm_state, conv_state) 或 None。
        ssm_state:[B,n_v,dk,dv] fp32 (None 则零初始化);
        conv_state:[B,K-1,qkv_dim] (与 x 同 dtype, None 则零初始化 → prefill 逐位不变)。
        返回 (attn_out [B,T,hidden], gdn_state)。"""
        B, T, _ = x.shape
        ssm_state, conv_state = gdn_state if gdn_state is not None else (None, None)
        qkv_mixed = self.wqkv(x)   # R11: f32 GEMM 输出
        if conv_state is None:
            # R11: 环 dtype 跟随 GEMM 输出 (f32) — 环存的是 qkv_mixed 卷积前原始行,
            # 必须与 conv 输入同 dtype (R10 环语义不变, 仅精度模式 bf16→f32)。
            conv_state = torch.zeros(B, self.conv_kernel - 1, self.qkv_dim,
                                     device=x.device, dtype=qkv_mixed.dtype)
        qkv_mixed, conv_state = self._conv1d(qkv_mixed, conv_state)   # (1) conv1d + silu + 环回写
        qkv_mixed = qkv_mixed.float()   # R11: conv 输出后 fp32 (L2norm/gate/beta/z 全程 f32)
        q, k, v = self._split_qkv(qkv_mixed, B, T)
        q = F.normalize(q, dim=-1, eps=1e-6)             # (2) q/k L2 归一
        k = F.normalize(k, dim=-1, eps=1e-6)
        q = q / math.sqrt(self.dk)                       # (2) q scale
        # GQA: tile 取模 n_k→n_v (R12 D3) — 对齐 llama.cpp qwen35.cpp:440-441 ggml_repeat_4d
        # (GGML_OP_REPEAT 取模语义: value head v → key head v%n_k), 非 block 复制 (v//n_k)。
        q = q.repeat(1, 1, self.gqa_ratio, 1)            # [B,T,n_v,dk] (4 因子匹配 4 维 [B,T,n_k,dk])
        k = k.repeat(1, 1, self.gqa_ratio, 1)
        z = self.wqkv_gate(x).reshape(B, T, self.n_v, self.dv)
        alpha_biased = self.ssm_alpha(x) + self.ssm_dt.to(x.dtype)   # [B,T,n_v]
        gate = F.softplus(alpha_biased) * self.ssm_a.to(x.dtype)     # (3) 负 decay 基
        beta = torch.sigmoid(self.ssm_beta(x))                     # (4) sigmoid
        if ssm_state is None:
            ssm_state = torch.zeros(B, self.n_v, self.dk, self.dv,
                                    device=x.device, dtype=torch.float32)
        o, ssm_state = self._recurrent(q, k, v, gate, beta, ssm_state, B, T)
        # R11: gated 保持 f32 直进 ssm_out GEMM — 删 .to(x.dtype) cast (对齐 llama.cpp
        # qwen35.cpp:454-461: ssm_norm 输出无 cast 直进 build_lora_mm(ssm_out, ...))。
        out = self.ssm_out(self._gated_norm(o, z))   # [B,T,hidden]
        return out, (ssm_state, conv_state)


# ---------------------------------------------------------------------------
# FullAttention (FA 在 il%4==3, GQA 24q/4kv, head_dim=256, MRoPE, gated output)
# ---------------------------------------------------------------------------
class FullAttention(nn.Module):
    """FA 层。输入 = 已归一化 hidden [B,T,hidden]。QG 联合投影 (q+gate), q/k 归一, MRoPE, GQA。"""

    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.hidden_size
        self.n_heads = cfg.num_attention_heads          # 24
        self.n_kv = cfg.num_key_value_heads             # 4
        self.head_dim = cfg.head_dim                    # 256
        self.rope_dim = cfg.rope_dim                    # 64
        self.mrope_section = cfg.mrope_section          # [11,11,10]
        self.rope_theta = cfg.rope_theta
        self.qg_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv * self.head_dim
        # GGUF 独立张量: attn_q [2*qg_dim, hidden] (q 前 qg_dim 行 + gate 后 qg_dim 行),
        # attn_k [kv_dim, hidden], attn_v [kv_dim, hidden], attn_output [hidden, qg_dim]
        self.wq = QuantLinear(hidden, 2 * self.qg_dim)
        self.wk = QuantLinear(hidden, self.kv_dim)
        self.wv = QuantLinear(hidden, self.kv_dim)
        self.wo = QuantLinear(self.qg_dim, hidden)
        self.attn_q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.attn_k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.use_gate = cfg.attn_output_gate

    def _apply_mrope(self, x, position_ids):
        """x:[B,T,n,d]。position_ids:[3,T] (t,h,w)。只对前 rope_dim 维旋转 (partial rotary)。
        Risk 1: section 和=32=rope_dim/2 (32 个旋转对, 每对 2 维); 第 32+ 对用轴 0。"""
        rope_dim = self.rope_dim
        d = x.shape[-1]
        x_rot, x_pass = x[..., :rope_dim], x[..., rope_dim:]
        n_pairs = rope_dim // 2
        sec = self.mrope_section
        axis = [0 if p < sec[0] else (1 if p < sec[0] + sec[1] else 2) for p in range(n_pairs)]
        axis = torch.tensor(axis, device=x.device)
        x1 = x_rot[..., 0::2]
        x2 = x_rot[..., 1::2]
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, rope_dim, 2, device=x.device).float() / rope_dim))
        # 每旋转对 p 选轴 axis[p]; pos[p, t] = position_ids[axis[p], t] → [n_pairs, T]
        T = x.shape[1]
        pos = position_ids[axis]                             # [n_pairs, T]
        freqs = pos.float() * inv_freq.reshape(n_pairs, 1)   # [n_pairs, T]
        cos = freqs.cos().t().view(1, T, 1, n_pairs)         # [1, T, 1, n_pairs]
        sin = freqs.sin().t().view(1, T, 1, n_pairs)
        out = torch.empty_like(x_rot)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x2 * cos + x1 * sin
        return torch.cat([out, x_pass], dim=-1)

    def _norm_head(self, x, norm):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm.eps)
        return (x.float() * rms * norm.weight.float()).to(x.dtype)

    def forward(self, x, kv_cache, position_ids):
        """x:[B,T,hidden] (已归一化)。返回 (attn_out [B,T,hidden], kv_cache)。"""
        B, T, _ = x.shape
        qg = self.wq(x)
        q, gate = qg[..., :self.qg_dim], qg[..., self.qg_dim:2 * self.qg_dim]
        k = self.wk(x)
        v = self.wv(x)
        q = q.view(B, T, self.n_heads, self.head_dim)
        gate = gate.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_kv, self.head_dim)
        v = v.view(B, T, self.n_kv, self.head_dim)
        q = self._norm_head(q, self.attn_q_norm)
        k = self._norm_head(k, self.attn_k_norm)
        q = self._apply_mrope(q, position_ids)
        k = self._apply_mrope(k, position_ids)
        # per-stage KV 按层累积 (self._layer_idx 由 decoder 层绑定)
        layer_idx = self._layer_idx
        if layer_idx in kv_cache:
            kc = kv_cache[layer_idx]
            kc["k"] = torch.cat([kc["k"], k], dim=1)
            kc["v"] = torch.cat([kc["v"], v], dim=1)
        else:
            kv_cache[layer_idx] = {"k": k, "v": v}
        kc = kv_cache[layer_idx]
        rep = self.n_heads // self.n_kv
        # R12 D3: block 复制 — 对齐 llama.cpp FA GQA (n_head_kv 原生 GQA, query head h → kv head h//rep,
        # rep = n_heads//n_kv; FA flash kernel ik2 = iq2 / rk2 是 block 除法, build_attn 无显式 repeat)。
        # kc["k"] 布局 [B, T, n_kv, head_dim] — head 轴是 dim 2 (非 dim 1), 故 repeat_interleave 取 dim=2。
        k_full = kc["k"].repeat_interleave(rep, dim=2)
        v_full = kc["v"].repeat_interleave(rep, dim=2)
        q_t = q.transpose(1, 2)
        k_t = k_full.transpose(1, 2)
        v_t = v_full.transpose(1, 2)
        if T > 1:
            mask = torch.ones(T, k_t.shape[2], dtype=torch.bool, device=x.device).tril().view(1, 1, T, -1)
            attn = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask)
        else:
            attn = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=False)
        attn = attn.transpose(1, 2).reshape(B, T, self.n_heads, self.head_dim)
        if self.use_gate:
            attn = attn * torch.sigmoid(gate)          # sigmoid gate 在 wo 之前
        attn = attn.reshape(B, T, self.qg_dim)
        out = self.wo(attn)
        return out, kv_cache


# 为 FullAttention.forward 提供 layer_idx (闭包注入) — 用属性绑定
def _bind_layer_idx(attn, layer_idx):
    attn._layer_idx = layer_idx


# ---------------------------------------------------------------------------
# MLP (PP variant: 纯 nn.Linear, 无 ColumnParallel/RowParallel — TP only, 禁)
# ---------------------------------------------------------------------------
class QwenMLP(nn.Module):
    """独立 ffn_gate / ffn_up / ffn_down (GGUF 不融合)。silu(gate)*up → down。"""

    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.hidden_size
        inter = cfg.intermediate_size
        self.inter = inter
        self.gate_proj = QuantLinear(hidden, inter)
        self.up_proj = QuantLinear(hidden, inter)
        self.down_proj = QuantLinear(inter, hidden)

    def forward(self, x):
        out = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(out)


# ---------------------------------------------------------------------------
# Decoder Layer — FM-003: 持有 input_layernorm + post_attention_layernorm
# ---------------------------------------------------------------------------
class Qwen3DecoderLayer(nn.Module):
    """input_layernorm(fused) → (GDN|FA) → residual → post_attention_layernorm → MLP → residual。
    CLONE-001: eager 路径零 clone。"""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.is_fa = (layer_idx % cfg.full_attention_interval == cfg.full_attention_interval - 1)
        # R12 D1: plain RMSNorm (去 ones_plus) — 对齐 llama.cpp qwen35.cpp:160 build_norm(LLM_NORM_RMS),
        # 公式 x*rsqrt(mean(x²)+eps)*w (w 为加载的原始 norm 权重, 非 1+w)。
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # R12 D1: plain RMSNorm (去 ones_plus) — 对齐 llama.cpp qwen35.cpp:187 build_norm(LLM_NORM_RMS)。
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if self.is_fa:
            self.attn = FullAttention(cfg)
            self.attn._layer_idx = layer_idx
        else:
            self.attn = GatedDeltaNet(cfg)
        self.mlp = QwenMLP(cfg)

    def forward(self, x, residual, kv_cache, gdn_states, position_ids):
        """x:[B,T,hidden], residual, kv_cache, gdn_states, position_ids:[3,T]。
        返回 (hs, res, kv_cache, gdn_states)。"""
        # FM-003: fused_add_rms_norm 必用本层 self.input_layernorm.weight
        normalized, res = self.input_layernorm(x, residual)
        if self.is_fa:
            attn_out, kv_cache = self.attn(normalized, kv_cache, position_ids)
        else:
            attn_out, gdn_state = self.attn(normalized, gdn_states.get(self.layer_idx))
            gdn_states[self.layer_idx] = gdn_state
        # FM-003: fused_add_rms_norm 必用本层 self.post_attention_layernorm.weight
        # 单流语义 (llama.cpp qwen35.cpp:180/:184): res = attn_out + res 恰好加一次,
        # res 即 post-attention 完整流 (ffn_residual); MLP 分支输出不掺残差 (:191),
        # 由下一层 input_layernorm 融合加 (:195) 或 PP 边界合并。
        mlp_in, res = self.post_attention_layernorm(attn_out, res)
        hs = self.mlp(mlp_in)
        return hs, res, kv_cache, gdn_states


# ---------------------------------------------------------------------------
# MTP block (blk.64) — 独立单元, 整体放末 stage
# ---------------------------------------------------------------------------
class MTPBlock(nn.Module):
    """MTP (nextn) blk.64 — 完整 FA decoder 层 (15 张量, FA-only):
      nextn.eh_proj [2h,h] (Q8_0), nextn.enorm/hnorm (F32)
      blk.64: attn_norm + attn_q/k/v/attn_output + attn_q_norm/k_norm (FA 张量)
              + post_attention_norm + ffn_gate/up/down
      → eh_proj(cat(enorm(token_emb), hnorm(hidden)))  # enorm 在前
        → input_layernorm → FA → residual → post_attention_layernorm → MLP → residual
        → shared_head_norm。"""

    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.hidden_size
        self.cfg = cfg
        self.eh_proj = QuantLinear(2 * hidden, hidden)
        self.enorm = RMSNorm(hidden, cfg.rms_norm_eps)
        self.hnorm = RMSNorm(hidden, cfg.rms_norm_eps)
        self.input_layernorm = RMSNorm(hidden, cfg.rms_norm_eps)
        self.attn = FullAttention(cfg)
        self.attn._layer_idx = cfg.mtp_layers[0] if cfg.mtp_layers else cfg.num_hidden_layers
        self.post_attention_layernorm = RMSNorm(hidden, cfg.rms_norm_eps)
        self.mlp = QwenMLP(cfg)
        self.shared_head_norm = RMSNorm(hidden, cfg.rms_norm_eps)

    def forward(self, hidden, token_emb, kv_cache, position_ids):
        # R11: 两路归一化输出必须同 dtype 才能 cat (生产流全 f32, 此 cast 为 no-op 防护 —
        # 合成自检 k6 的 token_emb 为 bf16 而 hidden 为 f32, 不防护则 cat 硬崩)。
        token_emb = token_emb.to(hidden.dtype)
        h = self.eh_proj(torch.cat([self.enorm(token_emb)[0], self.hnorm(hidden)[0]], dim=-1))  # enorm 在前 (llama.cpp qwen35.cpp: ggml_concat(e_norm, h_norm))
        normed, res = self.input_layernorm(h)
        out, kv_cache = self.attn(normed, kv_cache, position_ids)
        out = out + res
        normed, res = self.post_attention_layernorm(out)
        out = self.mlp(normed) + res
        out = self.shared_head_norm(out)[0]
        return out, kv_cache


# ---------------------------------------------------------------------------
# 顶层模型 (PP variant)
# ---------------------------------------------------------------------------
class Qwen3ForCausalLMTP(nn.Module):
    """Qwen3.5 hybrid 顶层 (PP variant)。首 stage 持 token_embd, 末 stage 持 LMHead。"""

    def __init__(self, cfg, pp_rank: int, pp_size: int, stage_layers: list[int], device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.stage_layers = stage_layers
        self.device = device
        self.is_first_rank = (pp_rank == 0)
        self.is_last_rank = (pp_rank == pp_size - 1)
        if self.is_first_rank:
            self.token_embd = QuantEmbedding(cfg.vocab_size, cfg.hidden_size)
        else:
            self.token_embd = None
        if self.is_last_rank:
            self.lm_head = QuantLinear(cfg.hidden_size, cfg.vocab_size)
        else:
            self.lm_head = None
        self.layers = nn.ModuleList()
        for il in stage_layers:
            self.layers.append(Qwen3DecoderLayer(cfg, il))
        self.mtp = MTPBlock(cfg) if (self.is_last_rank and cfg.mtp_layers) else None
        # R12 D1: plain RMSNorm (去 ones_plus) — 对齐 llama.cpp qwen35.cpp:206 build_norm(LLM_NORM_RMS, -1)。
        self.output_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, input_ids=None, hidden_states=None, residual=None,
                kv_cache=None, gdn_states=None, position_ids=None):
        g = get_pp_group()
        if input_ids is not None:
            B, T = input_ids.shape[0], input_ids.shape[1]
        else:
            B, T = hidden_states.shape[0], hidden_states.shape[1]
        token_emb_local = None  # 首 rank 输入 token 的 embedding (graph_mtp inp->embd)
        # 非首 stage: forward 前 recv 边界激活
        if not g.is_first_rank:
            it = g.recv_tensor_dict()
            hidden_states = it.tensors["hidden_states"]
            residual = it.tensors["residual"]
        # 首 stage: embedding (输出即输入 token 的 embedding, 层消费前捕获)
        if g.is_first_rank:
            # R11: QuantEmbedding 行 gather 反量化直出 f32 — 不再 cast bf16,
            # 生产流自入口起全程 f32 (对齐 llama.cpp F32 计算路径)。
            hidden_states = self.token_embd(input_ids)
            token_emb_local = hidden_states
            residual = None
        if position_ids is None:
            position_ids = torch.arange(T, device=hidden_states.device)
            position_ids = torch.stack([position_ids] * 3)
        kv_cache = {} if kv_cache is None else kv_cache
        gdn_states = {} if gdn_states is None else gdn_states
        for i, il in enumerate(self.stage_layers):
            hidden_states, residual, kv_cache, gdn_states = self.layers[i](
                hidden_states, residual, kv_cache, gdn_states, position_ids)
        if self.mtp is not None:
            # R5 (Issue 8): MTP 的 enorm 输入是**输入 token 的 embedding**, 不是 target 输出
            # hidden state — llama.cpp graph_mtp 双图输入 (qwen35.cpp:512 inp->embd /
            # :527 inp->h), token_emb 由 ggml_get_rows 从共享 embedding 权重行 gather
            # (:519-521)。PP 切层下 embedding 权重在首 rank、MTP 块在末 rank → token_emb
            # 必须跨 PP 边界:
            #   首 rank = 本地 embedding 输出 (token_emb_local, 层消费前的 alias; RMSNorm
            #   只分配新张量不原地改输入, alias 稳定 — rms_norm.py:53-72);
            #   其他 rank = 边界 token_embd 键 (pp.py IntermediateTensors 声明的缝)。
            # ⛔ 禁 self.token_embd(input_ids) (末 rank 为 None → R4 死点 TypeError) 与
            # else-hidden_states 兜底 (eh_proj 两路同为同一向量 → draft 数值必错)。
            token_emb = token_emb_local if g.is_first_rank else it.tensors.get("token_embd")
            if token_emb is None:
                raise PPError("PP-COMM-007: MTP 块需要输入 token embedding — 边界缺 "
                              "token_embd 键 (cfg.mtp_layers 非空时首 rank 必须打包)")
            # R8 单流: 修复 A 后 hidden_states 仅含末层 MLP 分支输出 — MTP 需完整累积流
            # (llama.cpp qwen35.cpp:527-531 的 inp->h / mtp_h_input, 进 hnorm)。
            final_stream = hidden_states + residual
            _, kv_cache = self.mtp(final_stream, token_emb, kv_cache, position_ids)
        if self.is_last_rank:
            normalized, _ = self.output_norm(hidden_states, residual)
            logits = self.lm_head(normalized)
            _logits_finiteness_diag(logits)  # ② 只读诊断, 不改数值路径
            return logits, kv_cache, gdn_states
        # 非末 stage: 打包边界激活并 send
        t = {
            "hidden_states": hidden_states.contiguous(),
            "residual": (residual if residual is not None else torch.zeros_like(hidden_states)).contiguous(),
        }
        # R5: 模型有 MTP 块 → 末 rank MTP 需要输入 token embedding, 首 rank 把自身 embedding
        # 输出打包为第 3 键 token_embd (shape (B,T,hidden) bf16); 无 MTP 则不发 (省边界流量)。
        if self.token_embd is not None and self.cfg.mtp_layers:
            t["token_embd"] = token_emb_local.contiguous()
        g.send_tensor_dict(IntermediateTensors(t))
        return None, kv_cache, gdn_states

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=0.0):
        """贪婪生成。PP 多 rank: 各 rank 锁步跑同环; 末 rank 采样出的 token 经 P2P
        传回首 rank (dst=(rank+1)%pp_size, rank1 回绕到 0), 首 rank 接收
        (src=(rank-1)%pp_size 回绕到末 rank) 后推进下一步输入 — 否则首 rank 会
        永远拿原 prompt 重复 prefill (R4 潜伏点: 本地 generated 在首 rank 恒空)。
        单 rank (pp_size==1 / dist 未初始化) 直通, 无 P2P。"""
        g = get_pp_group()
        multi = dist.is_initialized() and g.pp_size > 1
        generated = []
        cur = input_ids
        past_len = input_ids.shape[1]
        kv_cache, gdn_states = {}, {}
        for step in range(max_new_tokens):
            if cur.shape[1] > 1:
                # prefill: 整段, position_ids = arange(T) (forward 自动)
                position_ids = None
            else:
                # decode: 单 token, 位置 = past_len + (step-1)
                pos = past_len + step - 1
                position_ids = torch.stack([torch.tensor([pos], device=cur.device)] * 3)
            out = self.forward(input_ids=cur, kv_cache=kv_cache,
                               gdn_states=gdn_states, position_ids=position_ids)
            if g.is_last_rank:
                logits, kv_cache, gdn_states = out
                nxt = torch.argmax(logits[0, -1]).item() if temperature == 0.0 else _sample(logits, temperature)
                generated.append(nxt)
                cur = torch.tensor([[nxt]], device=cur.device, dtype=cur.dtype)
                if multi:
                    # R5: 采样 token → 首 rank (P2P, 回绕 dst=0)
                    tok = torch.tensor([nxt], device=cur.device, dtype=cur.dtype)
                    dist.send(tok, dst=g._norm_dst(None), group=g.device_group)
            else:
                _, kv_cache, gdn_states = out
                if multi and g.is_first_rank:
                    # R5: 首 rank ← 采样 token (P2P, 回绕 src=末 rank)
                    tok = torch.empty(1, device=cur.device, dtype=cur.dtype)
                    dist.recv(tok, src=g._norm_src(None), group=g.device_group)
                    nxt = int(tok.item())
                    generated.append(nxt)
                    cur = torch.tensor([[nxt]], device=cur.device, dtype=cur.dtype)
        return generated


def _sample(logits, temperature):
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()
