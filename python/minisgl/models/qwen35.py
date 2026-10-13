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
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from minisgl.core import Batch, get_global_ctx
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
        # Set by the engine before a spec-decode verification forward: a dict
        # into which the layer stores the per-token recurrence inputs
        # (qkv_mixed, gate, beta, alpha) so the GDN state can be re-advanced on
        # just the accepted prefix after rejection. None outside verification.
        self.verify_save: dict | None = None

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

        if self.verify_save is not None:
            # Save the recurrence inputs (post-conv q/k/v are derived from
            # qkv_mixed; the conv1d is re-run on the accepted prefix during
            # the state fix-up, so only the pre-conv mixed + gates are kept).
            self.verify_save["qkv"] = qkv_mixed.detach()  # [n, 10240]
            self.verify_save["gate"] = gate.detach()  # [n, 48]
            self.verify_save["beta"] = beta.detach()  # [n, 48]

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

    def rereun_recurrence(
        self,
        qkv_mixed: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        state: "GDNState",
    ) -> None:
        """Re-advance the GDN state on the accepted prefix (no GEMMs).

        After a spec-decode verification forward advances the state past the
        accepted prefix (including rejected drafts), this restores the state to
        the pre-verification snapshot and re-runs only the conv1d + delta-rule
        recurrence on the L accepted tokens. The projections are not re-run
        (their outputs were saved during the verification forward). Updates
        ``state.conv_buf`` and ``state.S`` in-place.
        """
        n = qkv_mixed.shape[0]
        if n == 0:
            return
        dev = qkv_mixed.device
        # Batched causal depthwise conv1d (kernel 4) + silu — the same math the
        # normal forward runs per chunk (conv over [3 prefix + n tokens] rows,
        # output row i = the i-th token), applied to the accepted prefix. The
        # per-token Python loop that used to be here launched ~12 tiny kernels
        # per token per layer (a pipeline drain on a launch-bound engine); the
        # kernel's n_tokens argument makes the whole prefix one launch.
        w = self.ssm_conv1d.view(self.conv_channels, 1, self.conv_kernel)  # [10240,1,4]
        conv_in = torch.cat([state.conv_buf, qkv_mixed], dim=0)  # [3+n, 10240]
        conv_in_t = conv_in.unsqueeze(0).transpose(1, 2)  # [1, 10240, 3+n]
        conv_out = F.conv1d(conv_in_t, w, groups=self.conv_channels)  # [1, 10240, n]
        # New conv_buf = the last (kernel-1) token rows of the concat — the
        # same update the normal forward does (it indexes the transposed view).
        # IN PLACE (copy_, not reassignment): if this layer ever runs under a
        # CUDA graph, the captured conv kernel holds a pointer to the live
        # buffer, and a reassignment would leave it reading stale memory.
        state.conv_buf.copy_(
            conv_in_t[0, :, -(self.conv_kernel - 1) :].transpose(0, 1).contiguous()
        )
        conv_out = F.silu(conv_out).squeeze(0).transpose(0, 1)  # [n, 10240]
        # Split q, k, v; L2-norm q, k (batched over the prefix)
        q = F.normalize(
            conv_out[:, : self.num_k * self.head_k].view(n, self.num_k, self.head_k),
            dim=-1,
            eps=1e-6,
        )
        k = F.normalize(
            conv_out[:, self.num_k * self.head_k : 2 * self.num_k * self.head_k].view(
                n, self.num_k, self.head_k
            ),
            dim=-1,
            eps=1e-6,
        )
        v = conv_out[:, 2 * self.num_k * self.head_k :].view(n, self.num_v, self.head_v)
        # One delta-rule recurrence over the whole accepted prefix
        # (n_tokens=n; the kernel is sequential over tokens internally).
        from minisgl.kernel import gdn as gdn_kernel

        S = state.S  # [48, 128, 128] float32
        q_f = q.float().unsqueeze(0).contiguous()  # [1, n, 16, 128]
        k_f = k.float().unsqueeze(0).contiguous()  # [1, n, 16, 128]
        v_f = v.float().unsqueeze(0).contiguous()  # [1, n, 48, 128]
        g_f = gate[:n].float().unsqueeze(0).contiguous()  # [1, n, 48]
        b_f = beta[:n].float().unsqueeze(0).contiguous()  # [1, n, 48]
        dst_f = torch.empty(1, n, self.num_v, self.head_v, device=dev, dtype=torch.float32)
        gdn_kernel.gdn_delta_net(
            q_f, k_f, v_f, g_f, b_f, dst_f, S,
            Hv=self.num_v, Hq=self.num_k,
            n_tokens=n, n_seqs=1,
            scale=self._scale,
        )


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
        # Total KV slots this rank's pool must hold (the MTP draft head, when
        # present, is assigned the next slot — see Qwen35ForCausalLM.__init__).
        self.num_kv_layers = local

    def reset_gdn_state(self, device: torch.device, dtype: torch.dtype):
        """Reset the per-request GDN recurrent state (call at sequence start).

        The GDN state is keyed by the *global* layer id so the kernel wrapper
        (which indexes ``ctx.gdn_state[layer_id]``) resolves correctly even
        when this rank only holds a slice of the layers.

        The tensors are allocated ONCE (first call) and zeroed in place on
        every later call. The CUDA graph captures the GDN kernel with these
        exact pointers, so they must stay fixed for the engine's lifetime:
        replacing them would leave the graph replaying against stale pointers
        and the GDN recurrent state would be disconnected from the prefill.
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

    def forward_with_pre_norm(
        self,
        h: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Back-rank forward that also returns the PRE-norm hidden state.

        Returns ``(normed_h, pre_norm_h)`` where ``pre_norm_h = h + residual``
        is captured before the final ``self.norm``. The MTP draft block consumes
        the pre-norm state (it applies its own ``hnorm``); the lm_head consumes
        the normed state.
        """
        assert self.is_back and not self.is_front
        import os as _os
        _lpt = _os.environ.get("SPEC_PHASE_TIMING") == "1"
        if _lpt:
            import time as _time
            import sys as _sys
            _t0 = _time.perf_counter()
        if _lpt:
            import time as _time
            import sys as _sys
            _slow = []
            _n = 0
            for layer in self.layers:
                _lt = _time.perf_counter()
                h, residual = layer.forward(h, residual)
                _lms = (_time.perf_counter() - _lt) * 1e3
                if _lms > 5.0:
                    _slow.append((_n, _lms, layer.is_full_attn))
                _n += 1
            if _slow:
                _det = " ".join(f"{i}:{m:.1f}ms{'/A' if a else '/G'}" for i, m, a in _slow)
                print(f"[layer-slow] {_det} (total slow={sum(m for _, m, _ in _slow):.1f}ms)",
                      file=_sys.stderr, flush=True)
        else:
            for layer in self.layers:
                h, residual = layer.forward(h, residual)
        if _lpt:
            _cpu_ms = (_time.perf_counter() - _t0) * 1e3
            print(f"[layer-cpu-ms] forward_with_pre_norm CPU wall={_cpu_ms:.1f}ms "
                  f"layers={len(self.layers)}", file=_sys.stderr, flush=True)
        pre_norm = h + residual
        normed = self.norm(pre_norm)
        return normed, pre_norm


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
      4. concat   = concat(e_norm, h_norm, dim=-1) [n, 2*5120]
      5. cur      = eh_proj @ concat               [n, 5120]
      6. inpSA    = cur  (residual)
      7. cur      = RMSNorm(cur, attn_norm)  (fused: returns (cur, inpSA))
      8. QG, K, V = full-attn projections (same as Qwen35FullAttn)
      9. cur      = attention(Q, K, V) * sigmoid(gate)
     10. cur      = attn_output @ cur
     11. cur      = cur + inpSA
     12. cur      = MLP(RMSNorm(cur, post_attention_norm)) + cur
     13. cur      = RMSNorm(cur, shared_head_norm)
     14. logits   = lm_head @ cur                  [n, vocab]
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
        # KV slot in the shared MHAKVCache pool; assigned by Qwen35ForCausalLM.
        self.kv_layer_id: int | None = None

    def forward(
        self,
        input_token: torch.Tensor,
        hidden_state: torch.Tensor,
        embed: Q80Embedding,
        lm_head: Q80LMHead,
        batch: "Batch",
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one MTP draft step with prefix attention over the MTP KV cache.

        Args:
            input_token: the sampled token from the previous step [n]
            hidden_state: the PRE-norm hidden state from the main model's last
                layer (h + residual, before the final RMSNorm) [n, 5120]. The
                block applies its own ``hnorm`` to it (matches llama.cpp
                graph_mtp, which feeds the unnormalized nextn input).
            embed: the main model's embedding (shared)
            lm_head: the main model's lm_head (shared)
            batch: the batch carrying ``out_loc`` and ``attn_metadata`` for the
                MTP KV slot.  ``out_loc`` points at the MTP slot's page-table
                entries for the current token positions; ``attn_metadata`` is
                built by ``prepare_metadata`` so the FlashAttention backend
                performs a prefix attention over the MTP KV cache.
            positions: per-token RoPE positions [n]. Defaults to
                ``batch.positions``.

        Returns:
            (logits, h_nextn): the logits for the next token [n, vocab] and the
            hidden state for the next MTP step [n, 5120].
        """
        ctx = get_global_ctx()
        n = input_token.shape[0]

        # 1. Embed the input token
        tok_embd = embed.forward(input_token)  # [n, 5120]

        # 2-3. Normalize the hidden state and the token embedding
        h_norm = self.hnorm(hidden_state)  # [n, 5120]
        e_norm = self.enorm(tok_embd)  # [n, 5120]

        # 4. Concatenate along the FEATURE dim (matches llama.cpp graph_mtp,
        # whose eh_proj weight is [5120, 10240]) and project
        concat = torch.cat([e_norm, h_norm], dim=-1)  # [n, 2*5120]
        cur = self.eh_proj(concat)  # [n, 5120]

        # 5-7. Input layernorm (RMSNormFused returns (normed, residual))
        cur, inpSA = self.input_layernorm(cur)  # cur [n, 5120], inpSA = pre-norm residual

        # 8. Full-attn projections
        qg = self.attn_q(cur)  # [n, 12288]
        qg = qg.view(n, self.num_qo, 2 * self.head_dim)  # [n, 24, 512]
        q = qg[:, :, : self.head_dim]  # [n, 24, 256]
        gate = qg[:, :, self.head_dim :]  # [n, 24, 256]

        q = self.attn_q_norm(q)  # [n, 24, 256]
        k = self.attn_k(cur).view(n, self.num_kv, self.head_dim)  # [n, 4, 256]
        k = self.attn_k_norm(k)  # [n, 4, 256]
        v = self.attn_v(cur)  # [n, kv_dim]

        # 9. Prefix attention over the dedicated MTP KV slot.  RoPE is applied
        # before the backend call (same order as the main full-attn layer); the
        # backend stores K/V at batch.out_loc and reads the full prefix through
        # batch.attn_metadata (cu_seqlens_k carries the MTP cached length).
        pos = positions if positions is not None else batch.positions
        q, k = self.rotary.forward(pos, q, k)
        o = ctx.attn_backend.forward(
            q, k.view(n, -1), v, self.kv_layer_id, batch
        )  # [n, 24, 256]
        o = o * torch.sigmoid(gate)  # [n, 24, 256]
        o = o.view(n, -1)  # [n, 6144]
        cur = self.attn_output(o)  # [n, 5120]

        # 10-11. Residual
        cur = cur + inpSA  # [n, 5120]

        # 12. MLP
        cur, ffn_residual = self.post_attention_layernorm(cur)  # cur = normed, ffn_residual = pre-norm
        gate = self.ffn_gate(cur)  # [n, ffn]
        up = self.ffn_up(cur)  # [n, ffn]
        cur = F.silu(gate) * up  # [n, ffn]
        cur = self.ffn_down(cur)  # [n, 5120]
        cur = cur + ffn_residual  # [n, 5120]

        # 13. Shared head norm
        cur = self.shared_head_norm(cur)  # [n, 5120]

        # 14. LM head.  reduce=False: this block needs one logit row per token
        # (populate fills every nextn slot; the draft chain scores every hop),
        # and the global ctx.batch is NOT this MTP batch — reduce=True would
        # gather rows by the outer batch's last indices and index out of bounds.
        logits = lm_head.forward(cur, reduce=False)  # [n, vocab]

        return logits, cur

    def _attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Single-step attention (no KV cache). q [n, 24, 256], k [n, 4, 256],
        v [n, 1024]. Returns [n, 24, 256].

        Each head carries exactly one query and one key/value (the current
        token — no sequence dim), so the per-head attention weight is the
        softmax of a single logit = 1 and the output is the GQA-expanded
        value vectors. (A multi-token bmm here is wrong: it would treat the
        256-dim head as 256 "positions" and mismatch the [n,24,24] scores
        against the [n,256,24] expanded values.)
        """
        n = q.shape[0]
        v = v.view(n, self.num_kv, self.head_dim).repeat_interleave(
            self.num_qo // self.num_kv, dim=1
        )  # [n, 24, 256]
        return v


@dataclass
class _MtpReq:
    """Synthetic decode Req for one MTP draft step (the draft head attends over
    the MTP KV cache, which already holds positions 0..draft_pos-1). A one-token
    decode at position ``draft_pos`` therefore has ``cached_len = draft_pos`` and
    ``device_len = draft_pos + 1`` (``extend_len`` 1). The attention backend's
    ``prepare_metadata`` reads only these four attributes, so this lightweight
    stand-in substitutes for a full ``Req``."""

    table_idx: int
    cached_len: int
    device_len: int

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len


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
        # The MTP draft head owns the KV slot immediately after the main model's
        # full-attn layers. The engine sizes the back-rank pool to num_kv_layers+1.
        if self.mtp is not None:
            self.mtp.kv_layer_id = self.model.num_kv_layers
        # The main model's embedding, copied to the back rank by the engine when
        # the MTP block is present (the draft block embeds its own tokens).
        # None until the engine performs the one-time copy.
        self.mtp_embed: Q80Embedding | None = None
        # Set by the engine after the communication group is set up.
        self.comm = None
        self.rank = 0
        self.world_size = 1
        self.dtype = torch.bfloat16

    def draft_mtp_chain(
        self,
        input_token: torch.Tensor,
        hidden_state: torch.Tensor,
        base_position: int,
        n_max: int,
        table_idx: int,
    ) -> torch.Tensor:
        """Run the MTP draft chain: n_max sequential single-row draft steps.

        Each draft step builds a synthetic decode batch so the attention backend
        performs prefix attention over the MTP KV slot: the batch's
        ``out_loc`` is the page-table entry for the draft position, and
        ``prepare_metadata`` computes ``cu_seqlens_k`` from the MTP cached
        length (base_position + i).  The MTP KV cache shares the main model's
        page table, so the pages for positions base_position..base_position+n_max-1
        must already be allocated (by the scheduler's verify-batch pre-alloc).

        Args:
            input_token: the sampled token from the main forward [1] int32.
            hidden_state: the pre-norm hidden state of that token [1, 5120].
            base_position: the RoPE position of ``input_token`` (t+1); the i-th
                draft token uses position t+1+i.
            n_max: number of draft tokens to produce.
            table_idx: the request's page-table row index.

        Returns:
            [n_max] int32 — the draft tokens d_1..d_n_max.
        """
        assert self.mtp is not None and self.mtp_embed is not None
        assert self.mtp.kv_layer_id is not None, (
            "MTP kv_layer_id not assigned — Qwen35ForCausalLM.__init__ must set it"
        )
        ctx = get_global_ctx()
        dev = input_token.device
        page_table = ctx.page_table  # [max_reqs, max_tokens] int32
        # The MTP KV cache was built over positions 0..base_position-1 by the
        # prefill MTP pass and the process() hook.  So before draft step i the
        # MTP cached length is base_position + i (positions 0..base_position+i-1).
        mtp_batch = Batch(reqs=[], phase="decode")
        mtp_batch.padded_reqs = []
        drafts = []
        cur_token = input_token  # [1]
        cur_hidden = hidden_state  # [1, 5120]
        # Draft positions are the contiguous range base_position-1 ..
        # base_position+n_max-2. torch.arange with device= generates them with
        # a small GPU kernel (start/end baked into the kernel args) — no
        # pageable H2D. The old per-hop torch.tensor([draft_pos], device=dev)
        # issued one synchronous H2D per hop (4 per step), parking the back
        # rank inside the draft chain.
        pos_all = torch.arange(
            base_position - 1, base_position - 1 + n_max,
            dtype=torch.int64, device=dev,
        )
        for i in range(n_max):
            # The MTP block at position t predicts token t+1 and builds the MTP KV
            # for position t, reading the prefix 0..t-1 (already built by the
            # prefill MTP pass plus the preceding draft steps). The prefill MTP
            # pass built positions 0..base_position-2, so the first draft step must
            # run at t = base_position-1: it reads prefix 0..base_position-2, writes
            # the KV at base_position-1, and predicts token base_position (the first
            # draft token). Hence draft_pos = base_position + i - 1.
            draft_pos = base_position + i - 1
            pos = pos_all[i : i + 1]  # [1] GPU view, no H2D
            # Synthetic Req carrying the MTP cached length for this draft step.
            mtp_req = _MtpReq(table_idx, draft_pos, draft_pos + 1)
            mtp_batch.reqs = [mtp_req]
            mtp_batch.padded_reqs = [mtp_req]
            mtp_batch.positions = pos
            mtp_batch.out_loc = page_table[table_idx, draft_pos].reshape(1)
            ctx.attn_backend.prepare_metadata(mtp_batch)
            logits, cur_hidden = self.mtp.forward(
                cur_token, cur_hidden, self.mtp_embed, self.lm_head,
                batch=mtp_batch, positions=pos,
            )
            next_token = torch.argmax(logits, dim=-1).to(torch.int32)  # [1]
            drafts.append(next_token)
            cur_token = next_token
        return torch.cat(drafts)  # [n_max]

    def _mtp_populate(self, input_ids: torch.Tensor, pre_norm: torch.Tensor, batch: Batch) -> None:
        """Build the MTP (nextn) KV cache in SHIFTED order.

        MTP slot s holds (token_{s+1}, h_s) -- the token one position AHEAD of
        the hidden state (llama.cpp graph_mtp / EAGLE3 nextn convention) -- so
        that slot s predicts token_{s+2} over the prefix s+1, which equals the
        real prefix. For the m tokens at positions p..p+m-1 this builds slots
        p..p+m-2 from (token_{p+i+1}, h_{p+i}), i = 0..m-2. Slot p+m-1 is left
        unwritten: its token does not exist yet (the draft chain's hop 0, or
        the verify batch's out_loc, writes it).

        A synthetic single-req batch stands in for the real batch: the MTP
        layer's attention reads only table_idx / cached_len / device_len /
        positions / out_loc, so _MtpReq substitutes for Req. Its own
        prepare_metadata slices the page table for exactly the slots being
        written ([: p+m-1]).
        """
        if self.mtp is None:
            return
        req0 = batch.reqs[0]
        m = input_ids.shape[0]
        if m < 2:
            return
        pos = batch.positions
        mtp_req = _MtpReq(req0.table_idx, req0.cached_len, req0.cached_len + m - 1)
        # phase="prefill": this is an m-1 token EXTEND (one query per slot),
        # not a one-token decode. The FA backend dispatches on phase — the
        # decode kernel would compute attention for only the first query and
        # leave the other m-2 rows of the output buffer unwritten (garbage).
        # With cached_len = 0 the metadata collapses to cu_seqlens_q ==
        # cu_seqlens_k, i.e. causal self-attention over exactly the slots
        # store_kv just wrote.
        mtp_batch = Batch(reqs=[mtp_req], phase="prefill")
        mtp_batch.padded_reqs = [mtp_req]
        mtp_batch.positions = pos[: m - 1]
        mtp_batch.out_loc = batch.out_loc[: m - 1]
        get_global_ctx().attn_backend.prepare_metadata(mtp_batch)
        self.mtp.forward(
            input_token=input_ids[1:],
            hidden_state=pre_norm[: m - 1],
            embed=self.mtp_embed,
            lm_head=self.lm_head,
            batch=mtp_batch,
            positions=pos[: m - 1],
        )

    def reset_gdn_state(self, device: torch.device, dtype: torch.dtype) -> None:
        """Delegate the per-request GDN recurrent-state reset to the inner model.

        The engine calls this on the outer model (``self.model``), but the GDN
        layers and their state live on the inner ``Qwen35Model``. Without this
        delegation the engine's ``hasattr`` guard is False and the startup reset
        is silently skipped, leaving ``ctx.gdn_state`` None and crashing the
        first forward pass on a GDN layer.
        """
        self.model.reset_gdn_state(device, dtype)

    _dump_step = 0

    def _maybe_dump(self, hidden: torch.Tensor, logits: torch.Tensor,
                    input_ids: torch.Tensor | None = None,
                    phase: str | None = None) -> None:
        if not os.environ.get("MINISGL_DUMP"):
            return
        max_steps = int(os.environ.get("MINISGL_DUMP_MAX", "8"))
        n = type(self)._dump_step
        type(self)._dump_step = n + 1
        if n >= max_steps:
            return
        ctx = get_global_ctx()
        batch = ctx.batch
        req_info = []
        for r in batch.reqs:
            req_info.append(f"req(cached={r.cached_len},dev={r.device_len},ext={r.extend_len})")
        padded_info = f"padded={len(batch.padded_reqs)}"
        print(f"[dump-debug] step{n} phase={phase} num_tokens={input_ids.shape[0] if input_ids is not None else -1} "
              f"reqs={batch.size} {req_info} {padded_info}", file=sys.stderr, flush=True)
        d = {"hidden": hidden.detach().cpu()}
        if input_ids is not None:
            d["input_ids"] = input_ids.detach().cpu()
            d["phase"] = phase
            d["num_tokens"] = int(input_ids.shape[0])
        if logits is not None:
            # logits is [num_tokens, vocab]; topk over the last dim -> [num_tokens, 5].
            # Record the LAST token's prediction (the one that emits next).
            topv, topi = torch.topk(logits.float(), 5)
            d["argmax"] = int(topi[-1][0].item())
            d["top5_tok"] = topi[-1].detach().cpu().tolist()
            d["top5_val"] = topv[-1].detach().cpu().tolist()
        torch.save(d, f"/tmp/engine_dump_step{n}.pt")

    def forward(self) -> torch.Tensor | None:
        ctx = get_global_ctx()
        if self.model.is_front and self.model.is_back:
            # No split: run all layers + the norm + the lm_head.
            output = self.model.forward(ctx.batch.input_ids)
            logits = self.lm_head.forward(output)
            self._maybe_dump(output, logits, ctx.batch.input_ids, ctx.batch.phase)
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
        if self.model.is_back:
            # Back rank: run the upper layers, capture the pre-norm hidden state.
            normed, pre_norm = self.model.forward_with_pre_norm(h=h, residual=residual)
            # Build the MTP KV cache in shifted nextn order: slot s holds
            # (token_{s+1}, h_s), the token one position AHEAD of the hidden
            # state. The draft chain's prefix attention reads this cache, so
            # it must be populated before the first draft step.
            # Guard on mtp_embed, not just mtp: the MTP block is constructed
            # from the GGUF's nextn layers (mtp_enabled), but the embedding
            # copy only runs when the engine is configured for spec decode
            # (mtp_n_max > 0). Direct-forward callers (the selftest) build the
            # model without that copy and must take the plain path.
            if (
                self.mtp is not None
                and self.mtp_embed is not None
                and ctx.batch.phase == "prefill"
            ):
                self._mtp_populate(ctx.batch.input_ids, pre_norm, ctx.batch)
            logits = self.lm_head.forward(normed)
            self._maybe_dump(normed, logits, ctx.batch.input_ids, ctx.batch.phase)
            return logits
        # Middle rank: the model returns (h, residual); hand off to the next rank.
        h_out, residual_out = self.model.forward(h=h, residual=residual)
        self._send_handoff(h_out, residual_out)
        return None

    def forward_compute(
        self,
        h: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        mtp: bool = False,
        reduce: bool = True,
    ):
        """Pure layer compute with no gloo handoff (CUDA-graph-capturable).

        The rank-to-rank activation handoff and the token relay are executed
        eagerly around this method (see the engine decode loop), so the captured
        region contains only device work. Returns:
          - front rank:  ``(h, residual)`` — the handoff activation;
          - back rank:   ``logits`` (consumes the provided ``h`` / ``residual``);
            with ``mtp=True`` and the MTP block present, ``(logits, pre_norm)``
            where ``pre_norm`` is the pre-final-norm hidden state the draft
            block consumes;
          - no split:    ``logits`` (embedding + all layers + lm_head).

        ``reduce`` is passed to the lm_head: True (default) collapses a prefill
        extend block to its last row (one logit row per request); False keeps
        every row (one logit row per token) — required by the spec-decode
        verification forward, which must score all K+1 draft tokens.
        """
        ctx = get_global_ctx()
        if self.model.is_front and self.model.is_back:
            return self.lm_head.forward(
                self.model.forward(ctx.batch.input_ids), reduce=reduce
            )
        if self.model.is_front:
            return self.model.forward(ctx.batch.input_ids)  # (h, residual)
        if mtp and self.mtp is not None:
            import os as _os
            import sys as _sys
            _fpt = _os.environ.get("SPEC_PHASE_TIMING") == "1"
            if _fpt:
                import torch as _t
                import time as _ftime
                _fe = [_t.cuda.Event(enable_timing=True) for _ in range(3)]
                # start_sync: drain whatever GPU work is ALREADY pending when
                # forward_compute begins (i.e. queued by the previous step).
                # If this blocks ~1.2 s, the stall is pending work from the
                # prior step; if it returns fast (<10 ms), the stall is queued
                # DURING forward_compute (between _fe[0] and the final sync).
                _st0 = _ftime.perf_counter()
                _t.cuda.synchronize()
                _start_sync_ms = (_ftime.perf_counter() - _st0) * 1e3
                _fe[0].record()
                # fc_cpu: CPU wall of the compute body only (from after
                # start_sync, i.e. the start of the real launch, through the
                # layers + lm_head, up to just before the final sync). If
                # fc_cpu ~ 325 ms while the fwdB window is ~1550 ms, the ~1.2 s
                # is GPU-idle / a hidden sync in an unmeasured gap; if fc_cpu ~
                # 1550 ms, the CPU is stalled in uninstrumented Python.
                _fc_t0 = _ftime.perf_counter()
            if _os.environ.get("SPEC_PROFILE") == "1":
                # Profile this single forward_with_pre_norm call and dump the
                # top CUDA kernels. Gated so it never runs in production. Only
                # profile every 4th step (the strict fast/slow alternation means
                # a sample of slow steps) and print only the top 8 kernels by
                # CUDA time, so the output is small.
                _spec_prof_count = getattr(self, "_spec_prof_count", 0) + 1
                self._spec_prof_count = _spec_prof_count
                if _spec_prof_count % 4 == 0:
                    from torch.profiler import profile as _tprof, ProfilerActivity as _tact
                    with _tprof(activities=[_tact.CUDA, _tact.CPU]) as _prof:
                        normed, pre_norm = self.model.forward_with_pre_norm(h=h, residual=residual)
                    _avgs = sorted(_prof.key_averages(), key=lambda e: e.self_device_time_total, reverse=True)
                    _n_tok = h.shape[0] if h is not None else -1
                    _lines = [f"[spec-profile] n={_n_tok} top-kernels-by-cuda-time:"]
                    for _ev in _avgs[:8]:
                        if _ev.self_device_time_total > 0:
                            _lines.append(f"  {_ev.key[:60]:60s} {_ev.self_device_time_total/1000:10.1f} ms")
                    print("\n".join(_lines), file=_sys.stderr, flush=True)
                else:
                    normed, pre_norm = self.model.forward_with_pre_norm(h=h, residual=residual)
            else:
                normed, pre_norm = self.model.forward_with_pre_norm(h=h, residual=residual)
            if _fpt:
                _fe[1].record()
                _lhead_t0 = _ftime.perf_counter()
            logits = self.lm_head.forward(normed, reduce=reduce)
            if _fpt:
                _lhead_cpu_ms = (_ftime.perf_counter() - _lhead_t0) * 1e3
                _fc_cpu_ms = (_ftime.perf_counter() - _fc_t0) * 1e3
                _fe[2].record()
                _sync_t0 = _ftime.perf_counter()
                _fe[2].synchronize()
                _sync_ms = (_ftime.perf_counter() - _sync_t0) * 1e3
                print(f"[fwdB-split-ms] layers={_fe[0].elapsed_time(_fe[1]):.1f} "
                      f"lm_head={_fe[1].elapsed_time(_fe[2]):.1f} "
                      f"lm_head_cpu={_lhead_cpu_ms:.1f} start_sync={_start_sync_ms:.1f} "
                      f"fc_cpu={_fc_cpu_ms:.1f} sync={_sync_ms:.1f}",
                      file=_sys.stderr, flush=True)
            return logits, pre_norm
        out = self.model.forward(h=h, residual=residual)  # normed hidden (back)
        return self.lm_head.forward(out, reduce=reduce)  # logits

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

    def send_tokens(self, tokens: torch.Tensor, tag: str = "") -> None:
        """Send the sampled tokens to the predecessor rank (rank - 1).

        The next tokens flow backward up the PP chain: rank N-1 -> N-2 -> ...
        -> 0. For a 2-rank split the predecessor is the front rank (rank 0).
        """
        t_cpu = tokens.detach().to("cpu")
        if os.environ.get("P2P_DEBUG"):
            print(f"[P2P] rank={self.rank} SEND {tag} n={t_cpu.numel()} "
                  f"vals={t_cpu.flatten()[:16].tolist()}", file=sys.stderr, flush=True)
        torch.distributed.send(t_cpu, dst=self.rank - 1, group=self.comm)

    def recv_tokens(self, n: int, tag: str = "") -> torch.Tensor:
        """Receive the sampled tokens from the successor rank (rank + 1)."""
        tokens_cpu = torch.empty((n,), dtype=torch.int32, device="cpu")
        torch.distributed.recv(tokens_cpu, src=self.rank + 1, group=self.comm)
        if os.environ.get("P2P_DEBUG"):
            print(f"[P2P] rank={self.rank} RECV {tag} n={tokens_cpu.numel()} "
                  f"vals={tokens_cpu.flatten()[:16].tolist()}", file=sys.stderr, flush=True)
        return tokens_cpu.to(torch.cuda.current_device())
