from __future__ import annotations

import os
import re
import time
from datetime import timedelta
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, get_global_ctx, set_global_ctx
from minisgl.distributed import (
    destroy_distributed,
    enable_pynccl_distributed,
    set_pp_info,
    set_tp_info,
)
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


# Matches a per-layer state_dict key: "model.layers.<global_id>.<rest>".
_LAYER_KEY_RE = re.compile(r"^(model\.layers\.)\d+(\..+)$")


def _remap_layer_key(key: str, start: int, end: int) -> Optional[str]:
    """Rewrite a global-indexed layer key to this rank's local layer index.

    The weight loader yields keys indexed by the *global* layer id, but a rank
    that holds only a slice of the layers indexes its own layers *locally*
    (0 .. split-1). Returns the locally-indexed key when the global layer id
    falls in ``[start, end)``, ``None`` when it is outside this rank's slice,
    and the key unchanged when it has no layer index (a global tensor such as
    the embedding, final norm, or lm_head — the caller's ``wanted`` set then
    decides whether this rank owns it).
    """
    m = _LAYER_KEY_RE.match(key)
    if not m:
        return key
    global_id = int(key[m.end(1): m.start(2)])
    if not (start <= global_id < end):
        return None
    return f"{m.group(1)}{global_id - start}{m.group(2)}"


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event
    # Spec-decode only: the per-request number of emitted tokens (L+1 each).
    # Empty for a normal single-token forward; non-empty when the scheduler
    # drove the two-forward spec step (forward A + verify B). The scheduler
    # splits next_tokens_gpu by these counts in _process_last_data.
    spec_emit_counts: tuple = ()


class Engine:
    def __init__(self, config: EngineConfig):
        # Expandable segments: on a GPU that is nearly full (e.g. a 21 GiB
        # model on a 22 GiB card), the caching allocator's fixed segments
        # fragment the free memory into small blocks after weight loading,
        # so a later large allocation (the 128 MiB attention workspace)
        # cannot find a contiguous run and OOMs even though ~180 MiB is
        # reserved-but-unallocated. Expandable segments grow a single
        # virtual region instead, keeping the free memory usable.
        # Must be set before the allocator's first init — the assert below
        # proves no CUDA op (hence no allocation) has run yet.
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        if config.uses_layer_split:
            # PP is orthogonal to TP: the communication world is the PP world,
            # but the sharding factor is 1 (each stage keeps the full head and
            # weight count of every layer it owns). Components read the PP info
            # via get_shard_size() / try_get_pp_info(), so set it here.
            set_pp_info(
                rank=config.tp_info.rank,
                size=len(config.layer_split),
                layer_split=config.layer_split,
            )
        _adjust_config(config)

        self.device = torch.device(config.device or f"cuda:{config.tp_info.rank}")
        self._wait_device_ready(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.config = config
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config, layer_range=config.layer_range)
        # Wire the point-to-point handoff (layer split) into the model. Each
        # rank only builds its own slice of layers; the front rank sends
        # (hidden, residual) to the back rank, which samples and sends the
        # next tokens back.
        self.model.comm = self.tp_cpu_group
        self.model.rank = config.tp_info.rank
        self.model.world_size = config.tp_info.size
        self.model.dtype = self.dtype
        self.model.load_state_dict(self._load_weight_state_dict(config))
        # Initialize the per-request GDN recurrent state (qwen35). The state
        # is a single global per-layer tensor updated in place by the GDN
        # kernel, so it carries the previous request's final recurrence into
        # the next request unless zeroed at the request boundary. The startup
        # reset covers the boot gate; the ctx hook covers every later request
        # (consumed by the next prefill forward — see Context.forward_batch).
        if hasattr(self.model, "reset_gdn_state"):
            self.model.reset_gdn_state(self.device, self.dtype)
            self.ctx.gdn_reset_fn = lambda: self.model.reset_gdn_state(self.device, self.dtype)

        # One-time copy of the front rank's Q80 embedding to the back rank.
        # The MTP draft block lives on the back rank (GPU1) and embeds its own
        # draft tokens, but the Q80 embedding matrix lives on the front rank
        # (GPU0). Copy the raw uint8 weights over the gloo CPU group once at
        # startup, after load_state_dict (the embedding must be loaded first);
        # the MTP block then reads self.model.mtp_embed for its token lookups.
        if config.mtp_n_max > 0 and config.uses_layer_split:
            from minisgl.layers.q80_linear import Q80Embedding

            vocab = config.model_config.vocab_size
            hidden = config.model_config.hidden_size
            if config.is_front_rank:
                # self.model is Qwen35ForCausalLM; .model is the inner
                # Qwen35Model; .embed is the front-only Q80Embedding.
                embed_weight = self.model.model.embed.weight  # [vocab, 544] uint8
                torch.distributed.send(
                    embed_weight.detach().to("cpu"),
                    dst=config.tp_info.rank + 1,
                    group=self.tp_cpu_group,
                )
            else:
                mtp_embed = Q80Embedding(vocab, hidden)
                buf = torch.empty_like(mtp_embed.weight, device="cpu")
                torch.distributed.recv(
                    buf, src=config.tp_info.rank - 1, group=self.tp_cpu_group
                )
                # weight is a plain tensor (BaseOP, not nn.Module): replace the
                # attribute directly with the received bytes on the device.
                mtp_embed.weight = buf.to(self.device)
                self.model.mtp_embed = mtp_embed

        # ======================= KV cache initialization ========================
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
            num_layers=(
                config.num_full_attn_layers
                + (1 if config.is_back_rank and config.mtp_n_max > 0 else 0)
                if config.uses_layer_split
                else None
            ),  # +1 KV slot for the MTP draft head (back rank only)
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        # Release reserved-but-unallocated memory from weight loading before the
        # attention backend allocates its workspace buffer. On a GPU that is
        # nearly full (e.g. a 21 GiB model on a 22 GiB card), the caching
        # allocator's retained segments can starve the next allocation.
        torch.cuda.empty_cache()
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.page_table_bytes = self.page_table.numel() * self.page_table.element_size()
        # The layer split's rank-to-rank gloo handoff (a blocking CPU op) is not
        # CUDA-graph-capturable, so disable graph capture on split ranks (the
        # decode path falls back to eager). This matches the B5 commit, which
        # produced clean no-spec decode output.
        graph_bs = [] if config.uses_layer_split else config.cuda_graph_bs
        graph_max_bs = None if config.uses_layer_split else config.cuda_graph_max_bs
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=graph_bs,
            cuda_graph_max_bs=graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )
        logger.info_rank0(
            "allocation table: weights=%s kv=%s activation_workspace=%s "
            "pinned_host=%s graph_pool=%s page_table=%s",
            mem_GB(self.weight_bytes),
            mem_GB(self.kv_bytes),
            mem_GB(0) + " (caching allocator, no separate pool)",
            mem_GB(0) + " (pinned per step, no startup pool)",
            mem_GB(self.graph_runner.graph_pool_bytes),
            mem_GB(self.page_table_bytes),
        )

    def _wait_device_ready(self, device: torch.device, timeout: float = 180.0) -> None:
        """Block until the CUDA device accepts work, retrying driver teardown.

        After the supervisor kills the incumbent origin, the driver mid-reset
        accepts context creation (torch.cuda.mem_get_info works) but a real
        H2D transfer or kernel launch raises 'device not ready' for ~30-60 s
        while it finishes tearing down the dead context. Probe with a small
        allocation + copy + kernel + synchronize and back off until it lands.
        """
        deadline = time.monotonic() + timeout
        start = deadline - timeout
        attempt = 0
        while True:
            attempt += 1
            try:
                torch.cuda.set_device(device)
                src = torch.empty(1024, dtype=torch.int32, pin_memory=True)
                dst = src.to(device)
                dst.add_(1)
                torch.cuda.synchronize(device)
                del dst, src
                if attempt > 1:
                    logger.info_rank0(
                        "device %s ready after %d retries (%.1f s)",
                        device,
                        attempt - 1,
                        time.monotonic() - start,
                    )
                return
            except Exception as e:  # noqa: BLE001 — the driver raises many forms
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"CUDA device {device} not ready after {timeout:.0f}s of retries"
                    ) from e
                wait = min(5.0, 0.5 * attempt)
                logger.info(
                    "device %s not ready yet (%s); retrying in %.1f s (attempt %d)",
                    device,
                    str(e).splitlines()[0][:120],
                    wait,
                    attempt,
                )
                time.sleep(wait)

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            if config.uses_layer_split:
                # The layer split moves activations rank-to-rank over the gloo
                # group (activation handoff, token exchange, the MIN
                # all_reduces). NCCL allreduce is a uniform-TP mechanism the
                # split never calls, so skip the pynccl init here (it links
                # libnccl, which this scenario does not need).
                logger.info_rank0(
                    "layer split: all cross-rank traffic uses gloo; pynccl (NCCL) not initialized"
                )
            else:
                max_bytes = (
                    config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
                )
                enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: (
                    torch.randint(
                        0, 256, size=v.shape, dtype=torch.uint8, device=self.device
                    )
                    if v.dtype == torch.uint8
                    else torch.randn_like(v, device=self.device)
                )
                for k, v in self.model.state_dict().items()
            }
        else:
            # uint8 tensors are raw Q6_K weight bytes; they must NOT be cast
            # to the model dtype (the Q6_K kernels dequantize in registers).
            #
            # The weight loader yields keys indexed by the *global* layer id,
            # but a rank holding only a slice of the layers indexes its own
            # layers *locally* (0 .. split-1). Remap each global layer id into
            # this rank's local range before filtering. Without this, a
            # non-front rank keeps global ids 0..split-1 instead of its own
            # slice: it loads the wrong layers and crashes on the first
            # GDN/full-attn type mismatch (e.g. KeyError on a full-attn
            # layer's attn_q when the slot holds a GDN layer's attn_qkv).
            start, end = config.layer_range
            wanted = set(self.model.state_dict().keys())
            out: Dict[str, torch.Tensor] = {}
            for k, v in load_weight(config.model_path, self.device):
                rk = _remap_layer_key(k, start, end)
                if rk is not None and rk in wanted:
                    out[rk] = v if v.dtype == torch.uint8 else v.to(self.dtype)
            return out

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        self.weight_bytes = max(0, old_free_memory - new_free_memory)
        if config.uses_layer_split:
            # Layer split: each rank holds its own full-attn layers and ALL of
            # the KV heads (no head sharding). Size the KV cache for this
            # rank's full-attn layers only.
            num_kv_layers = config.num_full_attn_layers
            num_kv_heads = config.model_config.num_kv_heads
        else:
            # TP: each rank holds a fraction of the KV heads and all layers.
            num_kv_layers = config.model_config.num_layers
            num_kv_heads = div_even(
                config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True
            )
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * num_kv_heads
            * config.page_size
            * self.dtype.itemsize
            * num_kv_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            available_memory = int(config.memory_ratio * old_free_memory) - self.weight_bytes
            num_pages = available_memory // cache_per_page
        # Sync across ranks: every rank must allocate the same page count so the
        # page table and the KV pool stay consistent across the split.
        num_pages_t = torch.tensor([num_pages], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            num_pages_t, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        num_pages = int(num_pages_t.item())
        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        self.kv_bytes = real_kv_size
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        # A non-uniform layer split deliberately puts different layer counts on
        # (possibly different-sized) GPUs, so cross-rank free memory is
        # imbalanced by design. The check only guards uniform TP.
        if (
            not self.config.uses_layer_split
            and max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024
        ):
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if self.config.uses_layer_split and self.config.is_front_rank:
                # Front rank: run the embedding + the lower layers + gloo handoff.
                self.model.forward()
                next_tokens_gpu = self.model.recv_tokens(batch.size, tag="D.nxt")
            elif self.config.uses_layer_split and self.config.is_back_rank:
                # Back rank: gloo handoff + upper layers + norm + lm_head + sample.
                logits = self.model.forward()
                next_tokens_gpu = self.sampler.sample(logits, args).to(torch.int32)
                self.model.send_tokens(next_tokens_gpu, tag="D.nxt")
            elif self.config.uses_layer_split:
                # Middle rank: receive the handoff from the predecessor, run
                # the layers this rank owns, and hand off to the successor
                # (the model's forward does the recv + run + send). Then relay
                # the sampled tokens from the successor back to the predecessor.
                self.model.forward()
                next_tokens_gpu = self.model.recv_tokens(batch.size, tag="D.rel")
                self.model.send_tokens(next_tokens_gpu, tag="D.rel")
            else:
                # No split: run all layers + the norm + the lm_head. Sample.
                if self.graph_runner.can_use_cuda_graph(batch):
                    logits = self.graph_runner.replay(batch)
                else:
                    logits = self.model.forward()
                # The lm_head already yields one row per request (get_last_indices
                # reduces the prefill extend block to its last row), so the
                # sampler consumes the full tensor as-is — no per-token gather.
                next_tokens_gpu = self.sampler.sample(logits, args).to(torch.int32)

        for req in batch.reqs:
            req.complete_one()

        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def forward_mtp_draft(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        """Spec-decode draft step: the normal decode forward (forward A) plus the
        MTP draft chain (back rank only).

        Forward A samples x_t (the token for position N+1, where N = cached_len).
        The back rank then runs the MTP block n_max times to draft the next n_max
        tokens (positions N+2 .. N+n_max+1). The front rank runs its compute via
        the CUDA graph (replay_front) when possible — the graph only yields
        (h, residual), which is all the front rank needs for forward A. The back
        rank runs the mtp=True path EAGERLY because the graph replay does not
        expose the pre-norm hidden state the draft chain consumes.

        Returns a ForwardOutput whose next_tokens_gpu holds the K+1 tokens
        [x_t, d_1, .., d_K] per request (row order = batch order), so the
        scheduler can scatter them into the token pool and build the verify
        batch. The per-request accepted length is not known until the verify
        forward; this method only produces the candidate tokens.
        """
        assert torch.cuda.current_stream() == self.stream
        n_max = self.config.mtp_n_max
        assert n_max > 0, "forward_mtp_draft requires mtp_n_max > 0"
        with self.ctx.forward_batch(batch):
            if self.config.uses_layer_split and self.config.is_front_rank:
                if self.graph_runner.can_use_cuda_graph(batch):
                    h, residual = self.graph_runner.replay_front(batch)
                else:
                    h, residual = self.model.forward_compute()
                self.model._send_handoff(h, residual)
                # Back rank replies with the [x_t, d_1..d_K] block in one
                # message: the D2H happens only after the draft chain's kernels
                # have been enqueued, so their launch time overlaps forward A's
                # GPU execution (two separate synchronous sends used to drain
                # the forward-A queue before the hops could even be launched).
                combined = self.model.recv_tokens((n_max + 1) * batch.size, tag="A.drafts")
                x_t_gpu = combined[: batch.size]  # [size]
                drafts_gpu = combined[batch.size :]  # [size*n_max]
            elif self.config.uses_layer_split and self.config.is_back_rank:
                assert self.model.mtp is not None, (
                    "forward_mtp_draft requires the MTP block on the back rank"
                )
                h, residual = self.model._recv_handoff(batch.input_ids.shape[0])
                # EAGER: the graph replay returns only logits, not the pre-norm
                # hidden state the draft chain consumes.
                logits, pre_norm = self.model.forward_compute(h, residual, mtp=True)
                x_t_gpu = self.sampler.sample(logits, args).to(torch.int32)  # [size]
                # Draft chain (back rank only): one token per request per hop.
                # base_position = position of x_t = cached_len + 1. The hops
                # enqueue behind forward A on the same stream, so their CPU
                # launch time overlaps forward A's GPU execution.
                drafts_gpu = torch.empty(
                    n_max * batch.size, dtype=torch.int32, device=self.device
                )
                for i, req in enumerate(batch.reqs):
                    base_position = req.cached_len + 1
                    d = self.model.draft_mtp_chain(
                        x_t_gpu[i : i + 1], pre_norm[i : i + 1], base_position, n_max,
                        req.table_idx,
                    )
                    drafts_gpu[i * n_max : (i + 1) * n_max] = d
                # One D2H + one gloo send for the [x_t, d_1..d_K] block (was
                # two synchronous sends: the first drained the forward-A queue
                # before the hops could even be launched).
                self.model.send_tokens(
                    torch.cat([x_t_gpu, drafts_gpu]), tag="A.drafts"
                )
            else:
                raise RuntimeError("forward_mtp_draft requires a 2-rank layer split")
        # Combine [x_t] and [d_1..d_K] into one [size, n_max+1] tensor row-major:
        # row i = [x_t_i, d_1_i, .., d_K_i]. The scheduler flattens to the
        # token pool as the K+1 verify tokens for request i.
        tokens_gpu = torch.empty(
            batch.size * (n_max + 1), dtype=torch.int32, device=self.device
        )
        # Reshape to [size, n_max+1] so the scatter is row-major: row i =
        # [x_t_i, d_1_i, .., d_K_i]. A strided slice like tokens_gpu[1::n_max+1]
        # would only select each row's d_1 slot (size `size`) and mismatch the
        # [size*n_max] drafts buffer.
        rows = tokens_gpu.view(batch.size, n_max + 1)
        rows[:, 0] = x_t_gpu
        rows[:, 1:] = drafts_gpu.view(batch.size, n_max)
        next_tokens_cpu = tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(tokens_gpu, next_tokens_cpu, copy_done_event)

    def forward_mtp_verify(
        self,
        verify_batch: Batch,
        args: BatchSamplingArgs,
    ) -> ForwardOutput:
        """Spec-decode verify step: the verification forward (forward B).

        verify_batch holds K+1 tokens per request at positions N+1 .. N+K+1
        ([x_t, d_1, .., d_K]), phase "prefill", reduce=False so the lm_head
        keeps every row (one logit row per draft token). The verification
        forward produces one logit row per token; the row from the token at
        position N+1+j predicts position N+2+j. On the back rank we accept the
        matching prefix of draft tokens and emit L+1 tokens (x_t + d_1..d_L).
        The bonus row (from d_K) is NEVER emitted by this step — it is the
        output of the NEXT forward A, which consumes d_L and generates N+L+2.

        GDN state fix-up: snapshot conv_buf + S of all 48 GDN layers BEFORE
        forward B (state = GDN(N)); forward B over-advances to GDN(N+K+1)
        (wrong when L<K); after acceptance restore the snapshot and re-run the
        conv1d + delta-rule recurrence on the first L verify inputs so the state
        is exactly GDN(N+L). The next forward A then consumes d_L.

        Returns a ForwardOutput whose next_tokens_gpu holds the L+1 emitted
        tokens per request (row order = batch order) and whose
        next_tokens_cpu is the same. The scheduler uses the per-request length
        to advance req.cached_len / req.device_len by L+1.
        """
        assert torch.cuda.current_stream() == self.stream
        n_max = self.config.mtp_n_max
        assert n_max > 0, "forward_mtp_verify requires mtp_n_max > 0"
        k_plus_1 = n_max + 1
        # NOTE: Batch.size is len(reqs), NOT the token count. The verify batch
        # holds K+1 tokens per request, so check the input_ids length.
        num_verify_tokens = verify_batch.input_ids.shape[0]
        assert num_verify_tokens == len(verify_batch.reqs) * k_plus_1, (
            f"verify batch must hold {k_plus_1} tokens per request, got "
            f"{num_verify_tokens} tokens for {len(verify_batch.reqs)} requests"
        )
        with self.ctx.forward_batch(verify_batch):
            if self.config.uses_layer_split and self.config.is_front_rank:
                # Forward B is a prefill-shaped forward (K+1 tokens per request)
                # so it is NOT graph-capturable (only decode sizes are captured).
                # The front rank runs its lower layers eagerly, hands off, and
                # waits for the back rank's acceptance verdict + emitted tokens.
                #
                # Forward B over-advances the FRONT rank's own GDN recurrent
                # state exactly like the back rank's (the front rank owns the
                # lower GDN layers, e.g. ids 0,1,2,4,5,6,... of the split). The
                # back-rank fix-up below restores only the back rank's state;
                # without a symmetric front-rank fix-up the next forward A (a
                # graph replay) consumes an over-advanced recurrent state and
                # emits garbage. Snapshot + re-run on the accepted prefix,
                # mirroring the back rank. The fix-up is only correct for one
                # in-flight request (the GDN state is one global tensor per
                # layer); the bench is latency-1, and the back rank already
                # asserts this invariant.
                ctx = get_global_ctx()
                snapshot = {}
                for layer in self.model.model.layers:
                    if not layer.is_full_attn:
                        st = ctx.gdn_state[layer.layer_id]
                        snapshot[layer.layer_id] = (
                            st.conv_buf.clone(),
                            st.S.clone(),
                        )
                for layer in self.model.model.layers:
                    if not layer.is_full_attn:
                        layer.attn.verify_save = {}
                h, residual = self.model.forward_compute()
                self.model._send_handoff(h, residual)
                # Back rank replies with the per-request accepted length, then
                # the flat emitted tokens (L_i+1 tokens for request i).
                accepted_gpu = self.model.recv_tokens(len(verify_batch.reqs), tag="B.acc")  # [size]
                total_emitted = int(accepted_gpu.sum().item()) + len(verify_batch.reqs)
                emitted_gpu = self.model.recv_tokens(total_emitted, tag="B.emit")
                # Front-rank GDN fix-up (mirror of the back rank's below):
                # restore the pre-forward-B snapshot, then re-advance the
                # recurrence on the first L0 accepted verify tokens.
                L0 = int(accepted_gpu[0].item())
                for layer in self.model.model.layers:
                    if not layer.is_full_attn:
                        st = ctx.gdn_state[layer.layer_id]
                        conv_buf, S = snapshot[layer.layer_id]
                        # IN PLACE: the front rank's forward A is a graph replay,
                        # so the captured pointer to st.conv_buf / st.S is baked
                        # in. Reassigning (conv_buf = clone()) would reintroduce
                        # the pointer-staleness bug fixed on the no-spec path;
                        # copy the pre-forward-B snapshot back into the live
                        # tensors instead. (The back rank reassigns safely
                        # because its forward A is eager.)
                        st.conv_buf.copy_(conv_buf)
                        st.S.copy_(S)
                        vs = layer.attn.verify_save
                        if L0 > 0:
                            layer.attn.rereun_recurrence(
                                vs["qkv"][:L0], vs["gate"][:L0], vs["beta"][:L0], st
                            )
                        layer.attn.verify_save = None
            elif self.config.uses_layer_split and self.config.is_back_rank:
                # PHASE TIMING (diagnostic only, SPEC_PHASE_TIMING=1): CUDA
                # events bracket each phase; elapsed_time is GPU-stream time.
                import os as _os
                _pt = _os.environ.get("SPEC_PHASE_TIMING") == "1"
                _evs = [torch.cuda.Event(enable_timing=True) for _ in range(8)] if _pt else None
                if _evs is not None:
                    _evs[0].record()
                assert self.model.mtp is not None, (
                    "forward_mtp_verify requires the MTP block on the back rank"
                )
                h, residual = self.model._recv_handoff(verify_batch.input_ids.shape[0])
                if _evs is not None:
                    _evs[1].record()
                # GDN snapshot BEFORE forward B (state = GDN(N)).
                ctx = get_global_ctx()
                snapshot = {}
                for layer in self.model.model.layers:
                    if not layer.is_full_attn:
                        st = ctx.gdn_state[layer.layer_id]
                        snapshot[layer.layer_id] = (
                            st.conv_buf.clone(),
                            st.S.clone(),
                        )
                # Enable per-token save in every GDN layer for the fix-up.
                for layer in self.model.model.layers:
                    if not layer.is_full_attn:
                        layer.attn.verify_save = {}
                if _evs is not None:
                    _evs[2].record()
                # Forward B: keep every draft token's logit row (reduce=False)
                # and the pre-norm hidden state (mtp=True) for the MTP rebuild.
                logits, pre_norm = self.model.forward_compute(h, residual, mtp=True, reduce=False)
                if _evs is not None:
                    _evs[3].record()
                # Rebuild the MTP (nextn) KV slots N+1..N+K with the true hidden
                # states: slot p = (token_{p+1}, h_p). The draft chain's slots are
                # stale (built from its own hiddens); the next chain reads this
                # cache as its prefix.
                self.model._mtp_populate(verify_batch.input_ids, pre_norm, verify_batch)
                if _evs is not None:
                    _evs[4].record()
                # logits: [size, vocab] = [size*(K+1), vocab]. Row for the token
                # at position N+1+j predicts position N+2+j. So the row for
                # x_t (j=0) predicts d_1; the row for d_i (j=i) predicts d_{i+1}.
                # Acceptance: v_j = argmax(logits[req_i * (K+1) + j]); accept d_j
                # (the j-th draft, j=1..K) iff v_{j-1} == d_j. L = first j in
                # 1..K where v_{j-1} != d_j, minus 1 (or K if all match).
                size = len(verify_batch.reqs)
                # The GDN recurrent state is a single GLOBAL per-layer tensor
                # (shared across requests), so the post-verify fix-up below is
                # only correct for one in-flight request. The bench is
                # latency-1; guard the invariant here.
                assert size == 1, (
                    "forward_mtp_verify GDN fix-up requires concurrency 1 "
                    f"(got {size} in-flight requests)"
                )
                # Acceptance, computed on GPU: the leading-True length of eq is
                # a cumsum/argmax, so the ONLY GPU->CPU sync in the verify step
                # is the single L read below. (The per-j bool(eq[j]) loop used
                # to issue one .item() per draft; the first one drained the
                # whole forward-B queue, and each later one re-synced.)
                base = 0
                d_rows = verify_batch.input_ids[base + 1 : base + 1 + n_max]  # [K]
                v_pred = logits[base : base + n_max].argmax(dim=-1)  # [K]
                eq = v_pred == d_rows  # [K] bool, on GPU
                # L = number of leading Trues of eq. cumprod stays 1 until the
                # first False, then 0 forever: sum(cumprod) is exactly the
                # leading-True count (all-True -> K, all-False -> 0). All GPU
                # kernels, no tensor creation — the old torch.where form built
                # torch.tensor(n_max, device=cuda) every step, a pageable H2D
                # that parked the back rank ~1.2 s/step (py-spy: 71% of samples
                # at engine.py:710).
                L_gpu = eq.to(torch.int32).cumprod(-1).sum()
                # The one sync of the step: read L. Everything forward B
                # produced (logits, MTP populate, GDN snapshot) completes here.
                L = int(L_gpu.item())
                if _evs is not None:
                    _evs[5].record()
                # --- spec token dump (diagnostic, env-gated) -------------------
                import os as _os_t
                if _os_t.environ.get("SPEC_TOKEN_DUMP") == "1":
                    global _SPEC_DUMP_N
                    try:
                        _SPEC_DUMP_N
                    except NameError:
                        _SPEC_DUMP_N = 0
                    if _SPEC_DUMP_N < 12:
                        _SPEC_DUMP_N += 1
                        import sys as _sys_t
                        _vids = verify_batch.input_ids[:base + 1 + n_max].tolist()
                        _vpre = v_pred.tolist()
                        _eqs = eq.tolist()
                        try:
                            _tok = load_tokenizer(self.config.model_path)
                            _vtxt = _tok.decode(_vids)
                            _ptxt = _tok.decode(_vpre)
                        except Exception as _e:
                            _vtxt = f"<decode-fail:{_e}>"
                            _ptxt = _vtxt
                        _d1_eq_xt = _vids[1] == _vids[0] if len(_vids) > 1 else None
                        print(
                            f"[spec-dump {_SPEC_DUMP_N}] L={L} d1==xt={_d1_eq_xt}\n"
                            f"  verify(x_t,d_1..d_K) ids={_vids}\n"
                            f"  verify text={_vtxt!r}\n"
                            f"  v_pred(argmax) ids={_vpre}\n"
                            f"  v_pred text={_ptxt!r}\n"
                            f"  eq={_eqs}",
                            file=_sys_t.stderr, flush=True,
                        )
                # ----------------------------------------------------------------
                accepted_cpu = torch.tensor([L], dtype=torch.int32)  # CPU
                # Emit x_t + d_1..d_L (L+1 tokens).
                emitted_gpu = verify_batch.input_ids[base : base + 1 + L].to(torch.int32)
                # Relay the acceptance verdict + emitted tokens to the front rank.
                self.model.send_tokens(accepted_cpu, tag="B.acc")
                self.model.send_tokens(emitted_gpu, tag="B.emit")
                if _evs is not None:
                    _evs[6].record()
                # GDN fix-up. The verify forward ALWAYS over-advances the global
                # GDN state to GDN(N+K+1) (it processes all K+1 verify tokens),
                # so the snapshot (GDN(N)) must ALWAYS be restored -- even when
                # L=0, otherwise the next forward A consumes a corrupt state and
                # emits garbage. The recurrence re-processes the accepted prefix
                # (x_t, d_1..d_{L-1}; L tokens) only when L>0; when L=0 the next
                # forward A re-processes x_t from the restored snapshot, so the
                # state is just GDN(N). With concurrency 1 the global GDN state
                # is the single request's state, so the fix-up is on the first L
                # tokens of the (single) verify batch.
                L0 = int(accepted_cpu[0].item())
                for layer in self.model.model.layers:
                    if not layer.is_full_attn:
                        st = ctx.gdn_state[layer.layer_id]
                        conv_buf, S = snapshot[layer.layer_id]
                        st.conv_buf = conv_buf.clone()
                        st.S.copy_(S)
                        vs = layer.attn.verify_save
                        if L0 > 0:
                            layer.attn.rereun_recurrence(
                                vs["qkv"][:L0], vs["gate"][:L0], vs["beta"][:L0], st
                            )
                        layer.attn.verify_save = None
                if _evs is not None:
                    _evs[7].record()
                    _evs[7].synchronize()
                    import sys as _sys
                    _names = ["recv", "snap+save", "fwdB", "mtp_pop", "accept+L", "send", "fixup"]
                    _ph = " ".join(
                        f"{_names[i]}={_evs[i].elapsed_time(_evs[i + 1]):.1f}" for i in range(7)
                    )
                    print(f"[phase-ms] {_ph} (L={L})", file=_sys.stderr, flush=True)
            else:
                raise RuntimeError("forward_mtp_verify requires a 2-rank layer split")
        next_tokens_cpu = emitted_gpu.to("cpu", non_blocking=True)
        if os.environ.get("ENGINE_RET_TRACE") == "1" and not self.config.is_back_rank:
            import sys as _sys
            torch.cuda.synchronize()
            _ids = next_tokens_cpu.tolist()
            print(f"[engine-ret N] ids={_ids} len={len(_ids)}", file=_sys.stderr, flush=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(emitted_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    if config.attention_backend == "auto":
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
