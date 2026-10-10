"""Qwen3.5 GDN-hybrid model (qwen35 arch).

64 decoder layers: 48 Gated DeltaNet (GDN) linear-attention layers and 16
full (gated) attention layers, interleaved with full-attn at indices
3, 7, 11, ..., 63 (every 4th layer starting at 3).

Tensor layout (from the GGUF, confirmed against llama.cpp b10927 qwen35.cpp):

GDN layer (48 layers, Q6_K weights):
    attn_qkv       [10240, 5120]   q/k/v mixed projection (d_inner + 2*num_k*head_k)
    attn_gate      [6144, 5120]    gate projection (d_inner)
    ssm_conv1d     [10240, 4]      depthwise causal conv (F32)
    ssm_dt.bias    [48]            dt bias (F32)
    ssm_a          [48]            decay rate, negative (F32)
    ssm_alpha      [48, 5120]      alpha projection (BF16)
    ssm_beta       [48, 5120]      beta projection (BF16)
    ssm_norm       [128]           per-v-head RMSNorm (F32)
    ssm_out        [5120, 6144]    output projection (Q6_K)
    ffn_gate/up    [17408, 5120]   MLP gate/up (Q6_K)
    ffn_down       [5120, 17408]   MLP down (Q6_K)
    attn_norm      [5120]          input RMSNorm (F32)
    post_attention_norm [5120]     post-attn RMSNorm (F32)

Full-attn layer (16 layers, Q6_K weights):
    attn_q         [12288, 5120]   QG projection (q + gate, 2 * 24 * 256)
    attn_k         [1024, 5120]    K projection (4 kv_heads * 256)
    attn_v         [1024, 5120]    V projection (4 kv_heads * 256)
    attn_q_norm    [256]           per-q-head RMSNorm (F32)
    attn_k_norm    [256]           per-k-head RMSNorm (F32)
    attn_output    [5120, 6144]    output projection (Q6_K)
    ffn_gate/up    [17408, 5120]   MLP gate/up (Q6_K)
    ffn_down       [5120, 17408]   MLP down (Q6_K)
    attn_norm      [5120]          input RMSNorm (F32)
    post_attention_norm [5120]     post-attn RMSNorm (F32)

Global:
    token_embd     [248320, 5120]  embedding (Q8_0)
    output         [248320, 5120]  lm_head (Q8_0)
    output_norm    [5120]          final RMSNorm (F32)

MTP block (blk.64, Q8_0 weights, lives on the back rank):
    attn_q/k/v, attn_output, ffn_gate/up/down, eh_proj (all Q8_0)
    token_embd and lm_head are shared with the main model (Q8_0)

Q6_K weights are kept as raw uint8 bytes (210 B per 256-element block) and
dequantized in registers by the Q6_K GEMV/GEMM/gather kernels, so every
weight byte is read from DRAM exactly once per row.

The GDN layers use a delta-rule recurrence with per-v-head state
S [num_v, head_v, head_k] = [48, 128, 128] and a causal conv buffer
[conv_kernel-1, 10240] = [3, 10240]. The full-attn layers use the
FlashInfer attention backend with a partial RoPE (n_rot=64 of head_dim=256).
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from minisgl.core import get_global_ctx
from minisgl.layers import (
    BaseOP,
    OPList,
    LinearReplicated,
    Q6KLinear,
    Q80Embedding,
    Q80LMHead,
    Q80Linear,
    RMSNorm,
    RMSNormFused,
)
from minisgl.models.base import BaseLLMModel


# ---------------------------------------------------------------------------
# GDN (Gated DeltaNet) layer
# ---------------------------------------------------------------------------


@dataclass
class GDNState:
    """Per-request recurrent state for one GDN layer."""

    conv_buf: torch.Tensor  # [conv_kernel-1, conv_channels] = [3, 10240]
    S: torch.Tensor  # [num_v, head_v, head_k] = [48, 128, 128]


class GDNLayer(BaseOP):
    """Gated DeltaNet linear-attention layer.

    Forward (from llama.cpp qwen35.cpp build_layer_attn_linear +
    delta-net-base.cpp build_delta_net_autoregressive):

      1. qkv_mixed = attn_qkv @ x          [n, 10240]
          z         = attn_gate @ x         [n, 6144]
          beta      = sigmoid(ssm_beta @ x) [n, 48]
          alpha     = ssm_alpha @ x         [n, 48]
          gate      = ssm_a * softplus(alpha + ssm_dt)          [n, 48]  (negative)
                      # ssm_a = -exp(A_log) (pre-transformed by the conversion
                      # script); HF: g = -A_log.exp()*softplus
      2. causal depthwise conv1d (kernel 4) on qkv_mixed, then silu
      3. split q [n,16,128], k [n,16,128], v [n,48,128]; L2-norm q,k
      4. repeat q,k from 16 k-heads to 48 v-heads
      5. delta-rule recurrence per token, per v-head:
            scale = 1/sqrt(head_k)
            S  = S * exp(gate)
            sk = S @ k
            d  = (v - sk) * beta
            S  = S + outer(d, k)   # S[value, key] += d[value] * k[key]
            o  = S @ (q * scale)
      6. out = RMSNorm(o, ssm_norm) * silu(z)
      7. final = ssm_out @ out             [n, 5120]
    """

    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size  # 5120
        d_inner = config.gdn_inner_size  # 6144
        head_k = config.gdn_state_size  # 128
        num_k = config.gdn_group_count  # 16
        num_v = config.gdn_time_step_rank  # 48
        head_v = d_inner // num_v  # 128
        conv_channels = d_inner + 2 * num_k * head_k  # 10240
        conv_kernel = config.gdn_conv_kernel  # 4

        self.d_inner = d_inner
        self.head_k = head_k
        self.num_k = num_k
        self.num_v = num_v
        self.head_v = head_v
        self.conv_channels = conv_channels
        self.conv_kernel = conv_kernel

        # Projections (Q6_K in the GGUF; kept as raw bytes)
        self.attn_qkv = Q6KLinear(hidden, conv_channels)
        self.attn_gate = Q6KLinear(hidden, d_inner)
        # BF16 in the GGUF (GGML type 30): the converter kept these two small
        # [48, 5120] projections unquantized. Plain F.linear with bfloat16
        # weights — the loader keeps them bfloat16 (no cast) and the
        # activation is bf16, so F.linear needs a same-dtype weight. 48x5120
        # GEMMs are compute-negligible next to the Q6_K GEMMs in the layer.
        self.ssm_beta = LinearReplicated(hidden, num_v, has_bias=False, dtype=torch.bfloat16)
        self.ssm_alpha = LinearReplicated(hidden, num_v, has_bias=False, dtype=torch.bfloat16)
        self.ssm_out = Q6KLinear(d_inner, hidden)

        # Conv (F32 in the GGUF)
        self.ssm_conv1d = torch.nn.Parameter(torch.empty(conv_channels, conv_kernel))

        # Scalars (F32 in the GGUF)
        self.ssm_dt = torch.nn.Parameter(torch.empty(num_v))
        self.ssm_a = torch.nn.Parameter(torch.empty(num_v))

        # Per-v-head RMSNorm (F32 in the GGUF)
        self.ssm_norm = RMSNorm(head_v, config.rms_norm_eps)

        self._scale = 1.0 / math.sqrt(head_k)

    def forward(self, x: torch.Tensor, state: GDNState) -> torch.Tensor:
        """x: [n_tokens, hidden]. Returns [n_tokens, hidden]."""
        n = x.shape[0]
        dev = x.device

        # 1. Projections
        qkv_mixed = self.attn_qkv(x)  # [n, 10240]
        z = self.attn_gate(x)  # [n, 6144]
        beta = torch.sigmoid(self.ssm_beta(x))  # [n, 48]
        alpha = self.ssm_alpha(x)  # [n, 48]
        # GDN gate: ssm_a * softplus(alpha + dt).  The GGUF stores the
        # pre-transformed -exp(A_log) (the conversion script applies
        # -torch.exp(A_log)), so this equals HF's g = -A_log.exp()*softplus.
        gate = F.softplus(alpha + self.ssm_dt) * self.ssm_a  # [n, 48]

        # 2. Causal depthwise conv1d (kernel 4) + silu
        #    conv_buf holds the (kernel-1) positions before this sequence.
        #    F.conv1d wants [N, C, L] (channels first): the kernel-1 prefix
        #    makes the n output positions exactly this sequence's tokens.
        conv_in = torch.cat([state.conv_buf, qkv_mixed], dim=0)  # [3+n, 10240]
        conv_in = conv_in.unsqueeze(0).transpose(1, 2)  # [1, 10240, 3+n]
        w = self.ssm_conv1d.view(self.conv_channels, 1, self.conv_kernel)  # [10240,1,4]
        conv_out = F.conv1d(conv_in, w, groups=self.conv_channels)  # [1, 10240, n]
        # update conv buffer to the last (kernel-1) positions
        state.conv_buf = conv_in[0, :, -(self.conv_kernel - 1) :].transpose(0, 1).clone()  # [3, 10240]
        conv_out = F.silu(conv_out).squeeze(0).transpose(0, 1)  # [n, 10240]

        # 3. Split q, k, v; L2-norm q, k along head_k
        q = conv_out[:, : self.num_k * self.head_k].view(n, self.num_k, self.head_k)
        k = conv_out[:, self.num_k * self.head_k : 2 * self.num_k * self.head_k].view(
            n, self.num_k, self.head_k
        )
        v = conv_out[:, 2 * self.num_k * self.head_k :].view(n, self.num_v, self.head_v)
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)

        # 4. Delta-rule recurrence (CUDA kernel; sequential over tokens,
        #    parallel across heads and state columns). q/k stay at Hq heads —
        #    the kernel maps v-head h to qk-head h % Hq (ggml_repeat_4d
        #    tiling, NOT repeat_interleave).
        S = state.S  # [48, 128, 128] float32
        o = torch.empty(n, self.num_v, self.head_v, device=dev, dtype=x.dtype)
        if n > 0:
            from minisgl.kernel import gdn as gdn_kernel

            # Reshape to [n_seqs=1, n_tokens, H, S_v] float32. The q/k/v
            # column-slices of conv_out are strided (row stride 10240), so
            # force contiguity — the kernel's TensorMatcher requires it.
            q_f = q.float().unsqueeze(0).contiguous()  # [1, n, 16, 128]
            k_f = k.float().unsqueeze(0).contiguous()  # [1, n, 16, 128]
            v_f = v.float().unsqueeze(0).contiguous()  # [1, n, 48, 128]
            g_f = gate.float().unsqueeze(0).contiguous()  # [1, n, 48]
            b_f = beta.float().unsqueeze(0).contiguous()  # [1, n, 48]
            dst_f = torch.empty(1, n, self.num_v, self.head_v, device=dev, dtype=torch.float32)

            gdn_kernel.gdn_delta_net(
                q_f, k_f, v_f, g_f, b_f, dst_f, S,
                Hv=self.num_v, Hq=self.num_k,
                n_tokens=n, n_seqs=1,
                scale=self._scale,
            )
            o = dst_f[0].to(x.dtype)  # [n, 48, 128]

        # 6. Gated output: per-v-head RMSNorm(o) * silu(z)
        o = o.view(-1, self.head_v)  # [n*48, 128]
        o = self.ssm_norm(o)  # per-128-dim-head RMSNorm
        o = o.view(n, self.d_inner)  # [n, 6144]
        o = o * F.silu(z)  # [n, 6144]

        # 7. Output projection
        return self.ssm_out(o)  # [n, 5120]


# ---------------------------------------------------------------------------
# Full (gated) attention layer
# ---------------------------------------------------------------------------


class _PartialRope:
    """Partial RoPE: rotate only the first ``rotary_dim`` dims of each head.

    For 1-D text positions, MRoPE with dimension_sections [11,11,10,0]
    (n_rot=64) degenerates to standard RoPE on the first 64 dims of each
    256-dim q/k head; the remaining 192 dims are passed through unchanged.

    The rotation pairs dim ``i`` with dim ``i + rotary_dim/2`` (i.e. (i, i+32)
    for i in [0, 32)), the same convention as llama.cpp's GGML_ROPE_TYPE_MROPE
    (``rotate_pairs`` with n_offset = n_dims/2) and HF's ``rotate_half``. The
    remaining dims [rotary_dim, head_size) are passed through unchanged.

    Implementation: flashinfer's Neox RoPE pairs dim i with i + head_size/2, so
    to get the (i, i+32) pairing we run it with head_size = rotary_dim (64) on
    the first 64 dims of each head and concatenate the untouched [64, 256) tail.

    Instances are shared across all full-attn layers via ``get_partial_rope``
    (one cos/sin cache per (head_size, rotary_dim, max_position, base)).
    """

    def __init__(self, head_size: int, rotary_dim: int, max_position: int, base: float):
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_pos, rotary_dim/2]
        cos = freqs.cos()
        sin = freqs.sin()
        # Real cos/sin for the rotary_dim/2 pairs. The RoPE op runs with
        # head_size = rotary_dim, so the cache is [max_pos, rotary_dim]
        # (cos | sin, no pass-through padding — the tail is handled separately).
        # Explicit float32: the engine builds the model under
        # set_default_dtype(bfloat16), and flashinfer requires an f32 cache.
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1).to(torch.float32)  # [max_pos, rotary_dim]

        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        self._apply = apply_rope_with_cos_sin_cache_inplace

    def forward(self, positions, q: torch.Tensor, k: torch.Tensor):
        # q: [n, num_qo_heads, head_size], k: [n, num_kv_heads, head_size]
        # Rotate the first rotary_dim dims of each head (head_size = rotary_dim
        # for the RoPE op, pairing (i, i + rotary_dim/2)); pass through the
        # remaining [rotary_dim, head_size) dims.
        n, hq, _ = q.shape
        n_, hk, _ = k.shape
        q_rot = q[..., : self.rotary_dim].contiguous().view(n, hq * self.rotary_dim)
        k_rot = k[..., : self.rotary_dim].contiguous().view(n_, hk * self.rotary_dim)
        self._apply(
            positions=positions,
            query=q_rot,
            key=k_rot,
            head_size=self.rotary_dim,
            cos_sin_cache=self._cos_sin_cache,
        )
        q_rot = q_rot.view(n, hq, self.rotary_dim)
        k_rot = k_rot.view(n_, hk, self.rotary_dim)
        q = torch.cat([q_rot, q[..., self.rotary_dim :]], dim=-1)
        k = torch.cat([k_rot, k[..., self.rotary_dim :]], dim=-1)
        return q, k


@functools.cache
def get_partial_rope(
    head_size: int, rotary_dim: int, max_position: int, base: float
) -> _PartialRope:
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope; the engine sets the shared rope
        # device (set_rope_device) before meta construction
        from minisgl.layers.rotary import get_rope_device

        device = get_rope_device()
        if device is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(device):
            return _PartialRope(head_size, rotary_dim, max_position, base)
    return _PartialRope(head_size, rotary_dim, max_position, base)


class Qwen35FullAttn(BaseOP):
    """Gated full-attention layer (Q6_K projections + FlashInfer seam).

    Forward (from llama.cpp qwen35.cpp build_layer_attn):

      1. QG = attn_q @ x                [n, 12288]
          per-head interleaved split (each 512-dim head = [256 Q | 256 gate]):
          q    = QG.view(n, 24, 512)[..., :256]    [n, 24, 256]
          gate = QG.view(n, 24, 512)[..., 256:]    [n, 24, 256]
      2. q = per-head RMSNorm(q, attn_q_norm [256])
      3. k = attn_k @ x                 [n, 4, 256]
          k = per-head RMSNorm(k, attn_k_norm [256])
      4. v = attn_v @ x                 [n, 4, 256]
      5. partial RoPE on q, k (n_rot 64, freq_base 1e7)
      6. o = attention(q, k, v)         [n, 24, 256]   (FlashInfer seam)
      7. o = o * sigmoid(gate)
      8. out = attn_output @ o          [n, 5120]
    """

    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size  # 5120
        head_dim = config.head_dim  # 256
        num_qo = config.num_qo_heads  # 24
        num_kv = config.num_kv_heads  # 4
        qo_dim = num_qo * head_dim  # 6144
        kv_dim = num_kv * head_dim  # 1024

        self.head_dim = head_dim
        self.num_qo = num_qo
        self.num_kv = num_kv

        # QG projection (q + gate, 2 * qo_dim)
        self.attn_q = Q6KLinear(hidden, 2 * qo_dim)  # [12288, 5120]
        self.attn_k = Q6KLinear(hidden, kv_dim)  # [1024, 5120]
        self.attn_v = Q6KLinear(hidden, kv_dim)  # [1024, 5120]
        self.attn_output = Q6KLinear(qo_dim, hidden)  # [5120, 6144]

        # Per-head RMSNorms
        self.attn_q_norm = RMSNorm(head_dim, config.rms_norm_eps)  # [256]
        self.attn_k_norm = RMSNorm(head_dim, config.rms_norm_eps)  # [256]

        # Partial RoPE (n_rot = 64 of head_dim = 256), shared across layers
        self.rotary = get_partial_rope(
            head_size=head_dim,
            rotary_dim=config.rotary_config.rotary_dim,  # 64
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,  # 1e7
        )

        # The KV pool is indexed by the local full-attn layer index. For a
        # layer split this rank's pool holds only this rank's full-attn layers,
        # so the model maps the global layer id to a local index. Default to
        # the global id (no split: the pool holds all layers).
        self.kv_layer_id: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        n = x.shape[0]

        # 1. QG projection + per-head interleaved split.
        #    llama.cpp (qwen35.cpp build_layer_attn): Qcur_full is [12288, n];
        #    Qcur  = view with head-stride 2*head_dim, offset 0      (first 256 of each 512-dim head)
        #    gate  = view with head-stride 2*head_dim, offset head_dim (second 256 of each 512-dim head)
        #    So within each 512-dim head: [0:256]=Q, [256:512]=gate. NOT a global split.
        qg = self.attn_q(x)  # [n, 12288]
        qg = qg.view(n, self.num_qo, 2 * self.head_dim)  # [n, 24, 512]
        q = qg[:, :, : self.head_dim]  # [n, 24, 256]
        gate = qg[:, :, self.head_dim :]  # [n, 24, 256]

        # 2. Per-head q norm
        q = self.attn_q_norm(q)

        # 3. K projection + per-head k norm
        k = self.attn_k(x).view(n, self.num_kv, self.head_dim)  # [n, 4, 256]
        k = self.attn_k_norm(k)

        # 4. V projection
        v = self.attn_v(x)  # [n, kv_dim]

        # 5. Partial RoPE
        q, k = self.rotary.forward(ctx.batch.positions, q, k)

        # 6. Attention (FlashInfer seam). Backend contract (scaffold
        # AttentionLayer): q is 3D [n, Hq, D]; k/v are 2D [n, Hkv*D] — the
        # store_cache kernel rejects 3D k/v.
        kv_id = self.kv_layer_id if self.kv_layer_id is not None else self.layer_id
        o = ctx.attn_backend.forward(q, k.view(n, -1), v, kv_id, ctx.batch)  # [n, 24, 256]

        # 7. Sigmoid gate
        o = o * torch.sigmoid(gate)

        # DEBUG: dump layer-3 intermediates for f64-ref comparison
        import os as _os
        if _os.environ.get("QWEN35_DEBUG_L3") and self.layer_id == 3:
            import torch as _t
            _d = _t.save if False else None
            _out = {
                "x": x.detach().float().cpu(),
                "qg": qg.detach().float().cpu(),
                "q": q.detach().float().cpu(),
                "k": k.detach().float().cpu(),
                "v": v.detach().float().cpu(),
                "gate": gate.detach().float().cpu(),
                "o": o.detach().float().cpu(),
                "positions": ctx.batch.positions.detach().cpu(),
            }
            _t.save(_out, "/tmp/bisect/eng_l3_inter.pt")
            print(f"[DEBUG] dumped layer-3 intermediates", flush=True)

        # 8. Output projection
        return self.attn_output(o.view(n, -1))  # [n, 5120]


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class Qwen35MLP(BaseOP):
    """Dense SiLU MLP (gate * up, then down)."""

    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size  # 5120
        ffn = config.intermediate_size  # 17408
        self.gate = Q6KLinear(hidden, ffn)  # [17408, 5120]
        self.up = Q6KLinear(hidden, ffn)  # [17408, 5120]
        self.down = Q6KLinear(ffn, hidden)  # [5120, 17408]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(x)  # [n, ffn]
        up = self.up(x)  # [n, ffn]
        x = F.silu(gate) * up  # [n, ffn]
        return self.down(x)  # [n, hidden]


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class Qwen35DecoderLayer(BaseOP):
    """One decoder layer: GDN or full-attn + MLP, with RMSNorm residuals."""

    def __init__(self, config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.is_full_attn = layer_id in config.full_attn_layer_ids
        if self.is_full_attn:
            attn = Qwen35FullAttn(config)
            attn.layer_id = layer_id
        else:
            attn = GDNLayer(config)
        self.attn = attn
        self.mlp = Qwen35MLP(config)
        self.input_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        if self.is_full_attn:
            x = self.attn.forward(x)
        else:
            state = get_global_ctx().gdn_state[self.layer_id]
            x = self.attn.forward(x, state)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Qwen35Model(BaseOP):
    """Embedding + decoder layers + final RMSNorm.

    With a non-uniform layer split, each rank builds only its own slice of
    layers (``layer_range``). The front rank (start == 0) owns the embedding;
    the back rank (end == num_layers) owns the final norm. ``forward`` returns
    the hidden state (back rank) or the (hidden, residual) handoff tuple
    (front rank).
    """

    def __init__(self, config, layer_range: tuple[int, int] | None = None):
        super().__init__()
        self.layer_range = layer_range or (0, config.num_layers)
        self.is_front = self.layer_range[0] == 0
        self.is_back = self.layer_range[1] == config.num_layers
        self.embed = (
            Q80Embedding(config.vocab_size, config.hidden_size) if self.is_front else None
        )
        self.layers = OPList(
            [
                Qwen35DecoderLayer(config, i)
                for i in range(self.layer_range[0], self.layer_range[1])
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps) if self.is_back else None
        self.config = config

        # Map each full-attn layer's global id to its local KV-pool index. The
        # pool holds only this rank's full-attn layers (a layer split), so the
        # local index is the ordinal among the full-attn layers in this range.
        local = 0
        for layer in self.layers:
            if layer.is_full_attn:
                layer.attn.kv_layer_id = local
                local += 1

    def reset_gdn_state(self, device: torch.device, dtype: torch.dtype):
        """Reset the per-request GDN recurrent state (call at sequence start).

        The GDN state is keyed by the *global* layer id so the kernel wrapper
        (which indexes ``ctx.gdn_state[layer_id]``) resolves correctly even
        when this rank only holds a slice of the layers.
        """
        ctx = get_global_ctx()
        ctx.gdn_state = {}
        for layer in self.layers:
            if not layer.is_full_attn:
                gdn = layer.attn
                ctx.gdn_state[layer.layer_id] = GDNState(
                    conv_buf=torch.zeros(
                        gdn.conv_kernel - 1, gdn.conv_channels, device=device, dtype=dtype
                    ),
                    S=torch.zeros(
                        gdn.num_v, gdn.head_v, gdn.head_k, device=device, dtype=torch.float32
                    ),
                )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ):
        if self.is_front:
            h = self.embed.forward(input_ids)  # [n, hidden]
            residual = None
        for layer in self.layers:
            h, residual = layer.forward(h, residual)
        if self.is_back:
            # The fused RMSNorm drops the final residual; add it back before
            # the final norm (matches the reference forward).
            h = h + residual
            h = self.norm(h)
            return h
        return h, residual


class MTPBlock(BaseOP):
    """MTP (multi-token-prediction) draft block.

    A single full-attention decoder layer plus a nextn head. It drafts the next
    token from (sampled token, hidden state) using the main model's embedding
    and lm_head. The MTP block lives on the back rank (GPU1) alongside the
    upper layers and the lm_head.

    Forward (from llama.cpp qwen35.cpp graph_mtp):
      1. tok_embd = embed(input_token)              [n, 5120]
      2. h_norm   = RMSNorm(hidden_state, hnorm)    [n, 5120]
      3. e_norm   = RMSNorm(tok_embd, enorm)        [n, 5120]
      4. concat   = concat(e_norm, h_norm, dim=0)   [2*5120, n]
      5. cur      = eh_proj @ concat                [5120, n]
      6. inpSA    = cur  (residual)
      7. cur      = RMSNorm(cur, attn_norm)
      8. QG, K, V = full-attn projections (same as Qwen35FullAttn)
      9. cur      = attention(Q, K, V) * sigmoid(gate)
     10. cur      = attn_output @ cur
     11. cur      = cur + inpSA
     12. cur      = MLP(RMSNorm(cur, post_attention_norm)) + cur
     13. cur      = RMSNorm(cur, shared_head_norm)
     14. logits   = lm_head @ cur                   [vocab, n]
    """

    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size  # 5120
        # Full-attn projections (Q8_0 for MTP block)
        head_dim = config.head_dim  # 256
        num_qo = config.num_qo_heads  # 24
        num_kv = config.num_kv_heads  # 4
        qo_dim = num_qo * head_dim  # 6144
        kv_dim = num_kv * head_dim  # 1024

        self.attn_q = Q80Linear(hidden, 2 * qo_dim)  # [12288, 5120]
        self.attn_k = Q80Linear(hidden, kv_dim)  # [1024, 5120]
        self.attn_v = Q80Linear(hidden, kv_dim)  # [1024, 5120]
        self.attn_output = Q80Linear(qo_dim, hidden)  # [5120, 6144]
        self.attn_q_norm = RMSNorm(head_dim, config.rms_norm_eps)  # [256]
        self.attn_k_norm = RMSNorm(head_dim, config.rms_norm_eps)  # [256]

        # MLP (Q8_0 for MTP block)
        ffn = config.intermediate_size  # 17408
        self.ffn_gate = Q80Linear(hidden, ffn)  # [17408, 5120]
        self.ffn_up = Q80Linear(hidden, ffn)  # [17408, 5120]
        self.ffn_down = Q80Linear(ffn, hidden)  # [5120, 17408]

        # Layer norms (F32)
        self.input_layernorm = RMSNormFused(hidden, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(hidden, config.rms_norm_eps)

        # Nextn head (F32 norms, Q8_0 eh_proj)
        self.eh_proj = Q80Linear(2 * hidden, hidden)  # [10240, 5120]
        self.enorm = RMSNorm(hidden, config.rms_norm_eps)  # [5120]
        self.hnorm = RMSNorm(hidden, config.rms_norm_eps)  # [5120]
        self.shared_head_norm = RMSNorm(hidden, config.rms_norm_eps)  # [5120]

        # Partial RoPE (n_rot = 64 of head_dim = 256), shared across layers
        self.rotary = get_partial_rope(
            head_size=head_dim,
            rotary_dim=config.rotary_config.rotary_dim,  # 64
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,  # 1e7
        )

        self.head_dim = head_dim
        self.num_qo = num_qo
        self.num_kv = num_kv

    def forward(
        self,
        input_token: torch.Tensor,
        hidden_state: torch.Tensor,
        embed: Q80Embedding,
        lm_head: Q80LMHead,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one MTP draft step.

        Args:
            input_token: the sampled token from the previous step [n]
            hidden_state: the hidden state from the main model's last layer [n, 5120]
            embed: the main model's embedding (shared)
            lm_head: the main model's lm_head (shared)

        Returns:
            (logits, h_nextn): the logits for the next token [vocab, n] and the
            hidden state for the next MTP step [n, 5120].
        """
        ctx = get_global_ctx()
        n = input_token.shape[0]

        # 1. Embed the input token
        tok_embd = embed.forward(input_token)  # [n, 5120]

        # 2-3. Normalize the hidden state and the token embedding
        h_norm = self.hnorm(hidden_state)  # [n, 5120]
        e_norm = self.enorm(tok_embd)  # [n, 5120]

        # 4. Concatenate and project
        concat = torch.cat([e_norm, h_norm], dim=0)  # [2*5120, n]
        cur = self.eh_proj(concat)  # [5120, n]

        # 5-6. Residual
        inpSA = cur

        # 7. Input layernorm
        cur = self.input_layernorm(cur)  # [5120, n]

        # 8. Full-attn projections
        qg = self.attn_q(cur)  # [n, 12288]
        qg = qg.view(n, self.num_qo, 2 * self.head_dim)  # [n, 24, 512]
        q = qg[:, :, : self.head_dim]  # [n, 24, 256]
        gate = qg[:, :, self.head_dim :]  # [n, 24, 256]

        q = self.attn_q_norm(q)  # [n, 24, 256]
        k = self.attn_k(cur).view(n, self.num_kv, self.head_dim)  # [n, 4, 256]
        k = self.attn_k_norm(k)  # [n, 4, 256]
        v = self.attn_v(cur)  # [n, kv_dim]

        # 9. Attention (no KV cache for MTP draft — single-step attention)
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        o = self._attention(q, k, v)  # [n, 24, 256]
        o = o * torch.sigmoid(gate)  # [n, 24, 256]
        o = o.view(n, -1)  # [n, 6144]
        cur = self.attn_output(o)  # [n, 5120]

        # 10-11. Residual
        cur = cur + inpSA  # [n, 5120]

        # 12. MLP
        ffn_residual = cur
        cur = self.post_attention_layernorm(cur)  # [n, 5120]
        gate = self.ffn_gate(cur)  # [n, ffn]
        up = self.ffn_up(cur)  # [n, ffn]
        cur = F.silu(gate) * up  # [n, ffn]
        cur = self.ffn_down(cur)  # [n, 5120]
        cur = cur + ffn_residual  # [n, 5120]

        # 13. Shared head norm
        cur = self.shared_head_norm(cur)  # [n, 5120]

        # 14. LM head
        logits = lm_head.forward(cur)  # [vocab, n]

        return logits, cur

    def _attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Single-step attention (no KV cache). q [n, 24, 256], k [n, 4, 256],
        v [n, 1024]. Returns [n, 24, 256]."""
        # Scale
        scale = 1.0 / math.sqrt(self.head_dim)
        # q [n, 24, 256], k [n, 4, 256] -> broadcast k to 24 heads
        k = k.repeat_interleave(self.num_qo // self.num_kv, dim=1)  # [n, 24, 256]
        v = v.view(-1, self.num_kv, self.head_dim).repeat_interleave(
            self.num_qo // self.num_kv, dim=1
        )  # [n, 24, 256]
        # Attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) * scale  # [n, 24, 256]
        # No mask (single-step)
        attn = torch.softmax(scores, dim=-1)  # [n, 24, 256]
        o = torch.bmm(attn, v.transpose(1, 2))  # [n, 24, 256]
        return o


class Qwen35ForCausalLM(BaseLLMModel):
    """Qwen3.5 GDN-hybrid causal LM (model + lm_head).

    With a non-uniform layer split, the front rank holds the embedding and the
    lower layers; the back rank holds the upper layers, the final norm, and the
    lm_head. The front rank hands off (hidden, residual) to the back rank; the
    back rank samples and sends the next tokens back to the front rank.

    When MTP is enabled (config.mtp_enabled), the back rank also holds the MTP
    draft block (blk.64) and the speculative decoding logic.
    """

    def __init__(self, config, layer_range: tuple[int, int] | None = None):
        super().__init__()
        self.model = Qwen35Model(config, layer_range=layer_range)
        self.lm_head = Q80LMHead(config.vocab_size, config.hidden_size) if self.model.is_back else None
        self.config = config
        # MTP draft block (only on the back rank when MTP is enabled)
        self.mtp = MTPBlock(config) if (config.mtp_enabled and self.model.is_back) else None
        # Set by the engine after the communication group is set up.
        self.comm = None
        self.rank = 0
        self.world_size = 1
        self.dtype = torch.bfloat16

    def reset_gdn_state(self, device: torch.device, dtype: torch.dtype) -> None:
        """Delegate the per-request GDN recurrent-state reset to the inner model.

        The engine calls this on the outer model (``self.model``), but the GDN
        layers and their state live on the inner ``Qwen35Model``. Without this
        delegation the engine's ``hasattr`` guard is False and the startup reset
        is silently skipped, leaving ``ctx.gdn_state`` None and crashing the
        first forward pass on a GDN layer.
        """
        self.model.reset_gdn_state(device, dtype)

    def _maybe_dump(self, hidden: torch.Tensor, logits: torch.Tensor) -> None:
        if not os.environ.get("MINISGL_DUMP"):
            return
        d = {"hidden": hidden.detach().cpu(), "logits": logits.detach().cpu()}
        if self.lm_head is not None:
            # Raw Q6_K bytes of the first 16 lm_head rows, so the f64 oracle
            # can byte-compare the engine's loaded weight against the GGUF.
            d["lm_head_w16"] = self.lm_head.weight[:16].detach().cpu()
        torch.save(d, "/tmp/engine_dump.pt")

    def forward(self) -> torch.Tensor | None:
        ctx = get_global_ctx()
        if self.model.is_front and self.model.is_back:
            # No split: run all layers + the norm + the lm_head.
            output = self.model.forward(ctx.batch.input_ids)
            logits = self.lm_head.forward(output)
            self._maybe_dump(output, logits)
            return logits
        if self.model.is_front:
            # Front rank: run the embedding + the lower layers. Hand off.
            h, residual = self.model.forward(ctx.batch.input_ids)
            self._send_handoff(h, residual)
            return None  # no logits on the front rank
        # Back or middle rank: receive the handoff, run the layers this rank owns.
        # The activation has one row per TOKEN (input_ids), not per request:
        # a prefill batch of 1 request can carry many tokens, so size the
        # receive buffer by the token count, not by batch.size (req count).
        h, residual = self._recv_handoff(ctx.batch.input_ids.shape[0])
        out = self.model.forward(h=h, residual=residual)
        if self.model.is_back:
            # Back rank: the model returns the final hidden state (norm applied).
            logits = self.lm_head.forward(out)
            self._maybe_dump(out, logits)
            return logits
        # Middle rank: the model returns (h, residual); hand off to the next rank.
        h_out, residual_out = out
        self._send_handoff(h_out, residual_out)
        return None

    def _send_handoff(self, h: torch.Tensor, residual: torch.Tensor) -> None:
        """Send (hidden, residual) to the successor rank (rank + 1).

        The activation flows forward down the PP chain: rank 0 -> 1 -> ... ->
        N-1. For a 2-rank split the successor is the back rank (world_size - 1),
        matching the original front->back handoff.
        """
        torch.distributed.send(h.detach().to("cpu"), dst=self.rank + 1, group=self.comm)
        torch.distributed.send(residual.detach().to("cpu"), dst=self.rank + 1, group=self.comm)

    def _recv_handoff(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Receive (hidden, residual) from the predecessor rank (rank - 1)."""
        device = torch.cuda.current_device()
        h_cpu = torch.empty((n, self.config.hidden_size), dtype=self.dtype, device="cpu")
        torch.distributed.recv(h_cpu, src=self.rank - 1, group=self.comm)
        residual_cpu = torch.empty((n, self.config.hidden_size), dtype=self.dtype, device="cpu")
        torch.distributed.recv(residual_cpu, src=self.rank - 1, group=self.comm)
        return h_cpu.to(device), residual_cpu.to(device)

    def send_tokens(self, tokens: torch.Tensor) -> None:
        """Send the sampled tokens to the predecessor rank (rank - 1).

        The next tokens flow backward up the PP chain: rank N-1 -> N-2 -> ...
        -> 0. For a 2-rank split the predecessor is the front rank (rank 0).
        """
        torch.distributed.send(tokens.detach().to("cpu"), dst=self.rank - 1, group=self.comm)

    def recv_tokens(self, n: int) -> torch.Tensor:
        """Receive the sampled tokens from the successor rank (rank + 1)."""
        tokens_cpu = torch.empty((n,), dtype=torch.int32, device="cpu")
        torch.distributed.recv(tokens_cpu, src=self.rank + 1, group=self.comm)
        return tokens_cpu.to(torch.cuda.current_device())
