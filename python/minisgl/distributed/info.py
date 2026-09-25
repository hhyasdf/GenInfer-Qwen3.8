from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


# ---------------------------------------------------------------------------
# Pipeline parallelism (PP)
#
# PP is orthogonal to TP. In this engine TP shards a single layer's heads and
# weights across GPUs; PP (a layer split) gives each GPU a *disjoint slice of
# the layers* and keeps the FULL head/weight count of every layer it owns.
# The two are never combined in this scenario (TP=1, PP=N), so the sharding
# factor is 1 in PP mode and the TP world size in TP mode.
#
# A layer split may be *asymmetric* (llama.cpp style): the per-rank layer
# counts need not be equal. ``layer_split`` is the tuple of per-rank counts
# that sums to the total layer count; rank r owns layers
# [sum(layer_split[:r]), sum(layer_split[:r+1])).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PPInfo:
    rank: int
    size: int
    layer_split: Tuple[int, ...] | None = None

    def __post_init__(self):
        assert 0 <= self.rank < self.size
        if self.layer_split is not None:
            assert len(self.layer_split) == self.size, (
                f"layer_split has {len(self.layer_split)} entries but PP size is {self.size}"
            )
            assert all(c > 0 for c in self.layer_split), (
                f"layer_split counts must be positive, got {self.layer_split}"
            )

    @property
    def uses_layer_split(self) -> bool:
        return self.layer_split is not None

    def layer_range(self, num_layers: int) -> Tuple[int, int]:
        """(start, end) global layer ids owned by this rank."""
        if self.layer_split is None:
            return (0, num_layers)
        start = sum(self.layer_split[: self.rank])
        end = start + self.layer_split[self.rank]
        return (start, end)


_PP_INFO: PPInfo | None = None


def set_pp_info(rank: int, size: int, layer_split: Tuple[int, ...] | None = None) -> None:
    global _PP_INFO
    if _PP_INFO is not None:
        raise RuntimeError("PP info has been set")
    _PP_INFO = PPInfo(rank, size, tuple(layer_split) if layer_split is not None else None)


def get_pp_info() -> PPInfo:
    if _PP_INFO is None:
        raise RuntimeError("PP info has not been set")
    return _PP_INFO


def try_get_pp_info() -> PPInfo | None:
    return _PP_INFO


def get_shard_size() -> int:
    """The head/weight sharding factor across ranks.

    In PP (layer-split) mode each stage keeps the full head/weight count of
    every layer it owns, so the factor is 1. In TP mode the factor is the TP
    world size (heads/weights are sharded across TP ranks). Components that
    derive local sizes from the full count must divide by this, not by the
    raw communication world size (which is the PP world size in PP mode).
    """
    pp = try_get_pp_info()
    if pp is not None and pp.uses_layer_split:
        return 1
    return get_tp_info().size


__all__ = [
    "DistributedInfo",
    "set_tp_info",
    "get_tp_info",
    "try_get_tp_info",
    "PPInfo",
    "set_pp_info",
    "get_pp_info",
    "try_get_pp_info",
    "get_shard_size",
]
