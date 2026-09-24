from __future__ import annotations

import re
from datetime import timedelta
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
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


class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _adjust_config(config)

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
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
        self.model.world_size = config.tp_info.size
        self.model.dtype = self.dtype
        self.model.load_state_dict(self._load_weight_state_dict(config))
        # Initialize the per-request GDN recurrent state (qwen35). It is reset
        # at sequence start; for a single-request engine the startup reset is
        # the one that matters for the boot gate.
        if hasattr(self.model, "reset_gdn_state"):
            self.model.reset_gdn_state(self.device, self.dtype)

        # ======================= KV cache initialization ========================
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
            num_layers=config.num_full_attn_layers if config.uses_layer_split else None,
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
        # With a layer split the rank-to-rank handoff is a blocking gloo
        # send/recv (a CPU op) and is not CUDA-graph-capturable, so disable
        # graph capture on split ranks (the decode path falls back to eager).
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
                # Front rank: run the embedding + the lower layers. The model's
                # forward hands off (hidden, residual) to the back rank and
                # returns None (no logits here). The back rank samples and
                # sends the next tokens back.
                self.model.forward()
                next_tokens_gpu = self.model.recv_tokens(batch.size)
            elif self.config.uses_layer_split and self.config.is_back_rank:
                # Back rank: receive the handoff, run the upper layers + the
                # norm + the lm_head, sample, and send the next tokens to the
                # front rank.
                logits = self.model.forward()
                next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
                self.model.send_tokens(next_tokens_gpu)
            else:
                # No split: run all layers + the norm + the lm_head. Sample.
                if self.graph_runner.can_use_cuda_graph(batch):
                    logits = self.graph_runner.replay(batch)
                else:
                    logits = self.model.forward()
                next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)

        for req in batch.reqs:
            req.complete_one()

        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

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
