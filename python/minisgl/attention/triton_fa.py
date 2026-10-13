"""Triton flash-attention backend (vLLM-pattern kernel; paged KV, GQA, causal).

Why this backend exists on this machine (recorded per the build-vs-buy
contract, see DESIGN.md):

- flashinfer FA2 batch-prefill (head_dim 256): "unspecified launch
  failure" on sm_75 for every layout/page_size/causality variant tested
  (fi_variants.py; logs/supervisor.log).
- sgl_kernel FA2/FA3: dispatch rejects sm_75 ("FlashAttention only
  supports Ampere GPUs or newer").
- cuDNN frontend SDPA: "SDPA FP16/BF16 requires SM80 (Ampere) or newer".
- torch SDPA (this build, 2.9.1+cu128): rejects enable_gqa for every head
  ratio and rejects q_len != k_len for every backend (the decode case is
  q_len=1, k_len=seqlen).

Triton is already a torch dependency (3.5.1) and compiles for sm_75. The
kernel follows the vLLM Triton-attention pattern (community
implementation): paged KV gather, GQA head mapping, causal masking with
bottom-right alignment, online (flash) softmax. FP32 accumulation
throughout; bf16 inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import triton
import triton.language as tl

from minisgl.core import Batch, get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@triton.jit
def _fa_prefill_kernel(
    Q,
    K,
    V,
    Out,
    cu_seqlens_q,
    seqused_k,
    page_table,
    sm_scale,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    stride_ptb,
    stride_ptp,
    Hq: tl.constexpr,
    Hkv: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Causal flash attention over paged KV, one request per program column.

    Grid: (cdiv(max_seqlen_q, BLOCK_M), bs * Hq). Query token m of request z
    sits at absolute position q_base + m (q_base = k_len - q_len) and
    attends to kv positions [0, q_base + m].
    """
    pid_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // Hq
    off_h = off_hz % Hq

    q_start = tl.load(cu_seqlens_q + off_z)
    q_len = tl.load(cu_seqlens_q + off_z + 1) - q_start
    if pid_m * BLOCK_M >= q_len:
        return
    k_len = tl.load(seqused_k + off_z)
    q_base = k_len - q_len  # absolute position of this request's first new query

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < q_len
    q = tl.load(
        Q + (q_start + offs_m)[:, None] * stride_qm + off_h * stride_qh + offs_d[None, :],
        mask=m_mask[:, None],
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q_abs = q_base + offs_m
    hi = tl.minimum(k_len, q_base + (pid_m + 1) * BLOCK_M)  # exclusive kv bound for this block

    kv_head = off_h // (Hq // Hkv)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < hi
        page = tl.load(page_table + off_z * stride_ptb + offs_n * stride_ptp, mask=n_mask, other=0)
        k = tl.load(
            K + page[:, None] * stride_kn + kv_head * stride_kh + offs_d[None, :],
            mask=n_mask[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        s = tl.where((offs_n[None, :] <= q_abs[:, None]) & n_mask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(
            V + page[:, None] * stride_vn + kv_head * stride_vh + offs_d[None, :],
            mask=n_mask[:, None],
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        Out + (q_start + offs_m)[:, None] * stride_om + off_h * stride_oh + offs_d[None, :],
        acc.to(Out.dtype.element_ty),
        mask=m_mask[:, None],
    )


@triton.jit
def _fa_decode_kernel(
    Q,
    K,
    V,
    Out,
    seqused_k,
    page_table,
    sm_scale,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    stride_ptb,
    stride_ptp,
    Hq: tl.constexpr,
    Hkv: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Single-query flash attention over paged KV (decode step).

    Grid: (bs, Hq). The query is the last token of the request and attends
    to kv positions [0, k_len).
    """
    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    k_len = tl.load(seqused_k + off_z)

    offs_d = tl.arange(0, D)
    q = tl.load(Q + off_z * stride_qm + off_h * stride_qh + offs_d)  # [D]

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    kv_head = off_h // (Hq // Hkv)
    for start_n in range(0, k_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < k_len
        page = tl.load(page_table + off_z * stride_ptb + offs_n * stride_ptp, mask=n_mask, other=0)
        k = tl.load(
            K + page[:, None] * stride_kn + kv_head * stride_kh + offs_d[None, :],
            mask=n_mask[:, None],
            other=0.0,
        )
        s = tl.sum(q.to(tl.float32)[None, :] * k.to(tl.float32), 1) * sm_scale  # [BLOCK_N]
        s = tl.where(n_mask, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha
        v = tl.load(
            V + page[:, None] * stride_vn + kv_head * stride_vh + offs_d[None, :],
            mask=n_mask[:, None],
            other=0.0,
        )
        acc += tl.sum(p[:, None].to(tl.float32) * v.to(tl.float32), 0)
        m_i = m_new

    acc = acc / l_i
    tl.store(Out + off_z * stride_om + off_h * stride_oh + offs_d, acc.to(Out.dtype.element_ty))


@dataclass
class TritonFAMetadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int
    page_table: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class TritonFABackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.scale = config.head_dim**-0.5
        self.capture: BaseCaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        # Tile sizes are bounded by shared memory per block:
        # sm_75 (Turing) 64 KB -> 32x32 tiles for D=256 (48 KB);
        # sm_80+ (Ada/Hopper) ~99 KB -> 64x32 tiles (64 KB).
        cc = torch.cuda.get_device_capability()
        if cc[0] >= 8:
            self.prefill_blocks = (64, 32)
            self.decode_block = 64
        else:
            self.prefill_blocks = (32, 32)
            self.decode_block = 32

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, TritonFAMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        out = torch.empty_like(q)
        D = self.config.head_dim
        Hq = self.config.num_qo_heads
        k_cache = self.kvcache.k_cache(layer_id).view(-1, self.config.num_kv_heads, D)
        v_cache = self.kvcache.v_cache(layer_id).view(-1, self.config.num_kv_heads, D)
        pt = metadata.page_table
        bs = batch.padded_size
        if batch.is_decode:
            _fa_decode_kernel[(bs, Hq)](
                q,
                k_cache,
                v_cache,
                out,
                metadata.cache_seqlens,
                pt,
                self.scale,
                q.stride(0),
                q.stride(1),
                k_cache.stride(0),
                k_cache.stride(1),
                v_cache.stride(0),
                v_cache.stride(1),
                out.stride(0),
                out.stride(1),
                pt.stride(0),
                pt.stride(1),
                Hq=Hq,
                Hkv=self.config.num_kv_heads,
                D=D,
                BLOCK_N=self.decode_block,
                num_warps=4,
            )
        else:
            BLOCK_M, BLOCK_N = self.prefill_blocks
            max_blocks = (metadata.max_seqlen_q + BLOCK_M - 1) // BLOCK_M
            _fa_prefill_kernel[(max_blocks, bs * Hq)](
                q,
                k_cache,
                v_cache,
                out,
                metadata.cu_seqlens_q,
                metadata.cache_seqlens,
                pt,
                self.scale,
                q.stride(0),
                q.stride(1),
                k_cache.stride(0),
                k_cache.stride(1),
                v_cache.stride(0),
                v_cache.stride(1),
                out.stride(0),
                out.stride(1),
                pt.stride(0),
                pt.stride(1),
                Hq=Hq,
                Hkv=self.config.num_kv_heads,
                D=D,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                num_warps=4,
            )
        return out

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS).to(device, non_blocking=True)
        cu_seqlens_k = (
            torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0).to(device, non_blocking=True)
        )
        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(0, len(reqs) + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):
            cu_seqlens_q = cu_seqlens_k
        else:
            cu_seqlens_q = (
                torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0).to(device, non_blocking=True)
            )
        page_table = get_global_ctx().page_table
        new_page_table = torch.stack(
            [page_table[req.table_idx, :max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")
        batch.attn_metadata = TritonFAMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = BaseCaptureData.create(max_bs, max_seq_len // self.page_size, self.kvcache.device)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = TritonFAMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, TritonFAMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)
