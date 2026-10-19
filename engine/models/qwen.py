# Phase 5(PP variant) + Phase 6 — Qwen3.5 hybrid model。
#
# 架构: 64 主层 (48 GDN 线性注意力 + 16 FullAttention, FA 在 il%4==3) + 1 MTP blk.64。
# norm 归属 (FM-003): decoder 层持有 input_layernorm + post_attention_layernorm,
#   fused_add_rms_norm 必用本层 weight; attention/GDN 层只做投影/recurrence/attention, 不再自归一。
# PP 变体: 每 stage 只建本区间层; MLP 用纯 nn.Linear (无 ColumnParallel/RowParallel — TP only, 禁)。
#   - FA 层: per-stage KV (按层切, 不按 head 切)。
#   - GDN 层: per-layer recurrent state [n_v, Dk, Dv] fp32, 归本 stage。
# 首 stage 持 token_embd, 末 stage 持 LMHead; 中间 stage 传 IntermediateTensors。
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels import RMSNorm
from ..pp import get_pp_group, IntermediateTensors, make_empty_intermediate_tensors
from ..weights.quant_linear import QuantLinear, QuantEmbedding


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
    GQA: 每 key head 挂 n_v/n_k 个 value head (repeat_interleave n_k→n_v)。
    Compute 修复 (Round-2):
      (1) conv1d depthwise + silu 作用于 qkv_mixed
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

    def _conv1d(self, qkv_mixed):
        """depthwise causal conv (kernel=conv_kernel) + silu。qkv_mixed [B,T,C] → [B,T,C]。"""
        B, T, C = qkv_mixed.shape
        w = self.ssm_conv1d.to(qkv_mixed.dtype)          # [C, K]
        out = qkv_mixed.new_zeros(B, T, C)
        out += qkv_mixed * w[:, 0]                        # j=0 全段 ([C] 沿 C 维广播)
        for j in range(1, self.conv_kernel):
            out[:, j:] += qkv_mixed[:, :T - j] * w[:, j]
        return F.silu(out)

    def _split_qkv(self, qkv_mixed, B, T):
        q = qkv_mixed[..., :self.key_dim].view(B, T, self.n_k, self.dk)
        k = qkv_mixed[..., self.key_dim:self.key_dim * 2].view(B, T, self.n_k, self.dk)
        v = qkv_mixed[..., self.key_dim * 2:].view(B, T, self.n_v, self.dv)
        return q, k, v

    def _recurrent(self, q, k, v, gate, beta, state, B, T):
        """q,k:[B,T,n_v,dk] (已 repeat_interleave) v:[B,T,n_v,dv] gate,beta:[B,T,n_v]。
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
                new_v = bt[:, j, None] * (vt[:, j] - sk)
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
        """x:[B,T,hidden] (已归一化)。gdn_state:[B,n_v,dk,dv] (None 则新分配)。
        返回 (attn_out [B,T,hidden], gdn_state)。"""
        B, T, _ = x.shape
        qkv_mixed = self.wqkv(x)
        qkv_mixed = self._conv1d(qkv_mixed)              # (1) conv1d + silu
        q, k, v = self._split_qkv(qkv_mixed, B, T)
        q = F.normalize(q, dim=-1, eps=1e-6)             # (2) q/k L2 归一
        k = F.normalize(k, dim=-1, eps=1e-6)
        q = q / math.sqrt(self.dk)                       # (2) q scale
        # GQA: repeat_interleave n_k→n_v
        q = q.repeat_interleave(self.gqa_ratio, dim=2)   # [B,T,n_v,dk]
        k = k.repeat_interleave(self.gqa_ratio, dim=2)
        z = self.wqkv_gate(x).reshape(B, T, self.n_v, self.dv)
        alpha_biased = self.ssm_alpha(x) + self.ssm_dt.to(x.dtype)   # [B,T,n_v]
        gate = F.softplus(alpha_biased) * self.ssm_a.to(x.dtype)     # (3) 负 decay 基
        beta = torch.sigmoid(self.ssm_beta(x))                     # (4) sigmoid
        if gdn_state is None:
            gdn_state = torch.zeros(B, self.n_v, self.dk, self.dv,
                                    device=x.device, dtype=torch.float32)
        o, gdn_state = self._recurrent(q, k, v, gate, beta, gdn_state, B, T)
        out = self.ssm_out(self._gated_norm(o, z).to(x.dtype))   # [B,T,hidden]
        return out, gdn_state


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
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, ones_plus=True)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, ones_plus=True)
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
        hs = attn_out + res
        # FM-003: fused_add_rms_norm 必用本层 self.post_attention_layernorm.weight
        mlp_in, res = self.post_attention_layernorm(hs, None)
        mlp_out = self.mlp(mlp_in)
        hs = mlp_out + res
        return hs, res, kv_cache, gdn_states


# ---------------------------------------------------------------------------
# MTP block (blk.64) — 独立单元, 整体放末 stage
# ---------------------------------------------------------------------------
class MTPBlock(nn.Module):
    """MTP (nextn) blk.64 — 完整 FA decoder 层 (15 张量, FA-only):
      nextn.eh_proj [2h,h] (Q8_0), nextn.enorm/hnorm (F32)
      blk.64: attn_norm + attn_q/k/v/attn_output + attn_q_norm/k_norm (FA 张量)
              + post_attention_norm + ffn_gate/up/down
      → eh_proj(cat(hnorm(hidden), enorm(token_emb)))
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
        h = self.eh_proj(torch.cat([self.hnorm(hidden)[0], self.enorm(token_emb)[0]], dim=-1))
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
        self.output_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, ones_plus=True)

    def forward(self, input_ids=None, hidden_states=None, residual=None,
                kv_cache=None, gdn_states=None, position_ids=None):
        g = get_pp_group()
        if input_ids is not None:
            B, T = input_ids.shape[0], input_ids.shape[1]
        else:
            B, T = hidden_states.shape[0], hidden_states.shape[1]
        # 非首 stage: forward 前 recv 边界激活
        if not g.is_first_rank:
            it = g.recv_tensor_dict()
            hidden_states = it.tensors["hidden_states"]
            residual = it.tensors["residual"]
        # 首 stage: embedding
        if g.is_first_rank:
            hidden_states = self.token_embd(input_ids).to(torch.bfloat16)
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
            token_emb = self.token_embd(input_ids).to(torch.bfloat16) if input_ids is not None else hidden_states
            _, kv_cache = self.mtp(hidden_states, token_emb, kv_cache, position_ids)
        if self.is_last_rank:
            normalized, _ = self.output_norm(hidden_states, residual)
            logits = self.lm_head(normalized)
            return logits, kv_cache, gdn_states
        # 非末 stage: 打包边界激活并 send
        it = IntermediateTensors({
            "hidden_states": hidden_states.contiguous(),
            "residual": (residual if residual is not None else torch.zeros_like(hidden_states)).contiguous(),
        })
        g.send_tensor_dict(it)
        return None, kv_cache, gdn_states

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=0.0):
        g = get_pp_group()
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
                position_ids = torch.stack([torch.tensor([pos])] * 3)
            out = self.forward(input_ids=cur, kv_cache=kv_cache,
                               gdn_states=gdn_states, position_ids=position_ids)
            if g.is_last_rank:
                logits, kv_cache, gdn_states = out
                nxt = torch.argmax(logits[0, -1]).item() if temperature == 0.0 else _sample(logits, temperature)
                generated.append(nxt)
                cur = torch.tensor([[nxt]], device=cur.device, dtype=cur.dtype)
                # PP 多 rank: 末 rank 需把采样 token 传回首 rank (P2P, dst=0)
                # 单 rank (pp_size=1) 无此步。多 rank 协调为 E2E/Verification 职责。
            else:
                _, kv_cache, gdn_states = out
                # 非末 stage: 跟随已生成 token (多 rank 需从末 rank recv)
                if generated:
                    cur = torch.tensor([[generated[-1]]], device=cur.device, dtype=cur.dtype)
        return generated


def _sample(logits, temperature):
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()
