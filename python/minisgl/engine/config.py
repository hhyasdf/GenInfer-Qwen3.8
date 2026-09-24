from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    # Gloo handoff timeout. Must cover the first request's one-time JIT
    # compiles (flashinfer rope ~45 s + prefill kernel ~30 s on a cold cache)
    # or the peer rank times out mid-compile and kills the run.
    distributed_timeout: float = 600.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # Non-uniform layer split across ranks (activation handoff, not TP). When
    # set, ``tp_info.size`` must equal ``len(layer_split)`` and rank i runs
    # main-model layers ``[sum(layer_split[:i]), sum(layer_split[:i+1]))``.
    # The front rank (rank 0) owns the embedding; the back rank (the last
    # rank) owns the final norm + lm_head and produces the logits.
    layer_split: Tuple[int, ...] | None = None
    # The cuda device index that hosts the draft model (speculative decoding).
    draft_device: int | None = None

    @property
    def uses_layer_split(self) -> bool:
        return self.layer_split is not None

    @property
    def layer_range(self) -> Tuple[int, int]:
        """The (start, end) main-model layer indices this rank runs."""
        if self.layer_split is None:
            return (0, self.model_config.num_layers)
        start = sum(self.layer_split[: self.tp_info.rank])
        end = start + self.layer_split[self.tp_info.rank]
        return (start, end)

    @property
    def is_front_rank(self) -> bool:
        """Rank 0 owns the embedding (or all ranks, when there is no split)."""
        return self.layer_split is None or self.tp_info.rank == 0

    @property
    def is_back_rank(self) -> bool:
        """The last rank owns the final norm + lm_head (or all ranks, no split)."""
        return self.layer_split is None or self.tp_info.rank == len(self.layer_split) - 1

    @property
    def num_full_attn_layers(self) -> int:
        """Full-attn layers this rank runs (for per-rank KV-cache sizing)."""
        start, end = self.layer_range
        return sum(1 for i in range(start, end) if i in self.model_config.full_attn_layer_ids)

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig
        from minisgl.models.gguf_weight import is_gguf_model_path

        if is_gguf_model_path(self.model_path):
            from minisgl.models.gguf_reader import GgufReader
            from minisgl.models.gguf_weight import find_main_gguf

            main_shard = find_main_gguf(self.model_path)
            with GgufReader(main_shard) as reader:
                return ModelConfig.from_gguf(reader.metadata)
        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
