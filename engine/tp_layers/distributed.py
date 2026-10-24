# Phase 2 — TP 通信原语 (tp_communication_contracts.md)。
#
# 3 种 collective: all_reduce_sum, all_gather_last_dim, CustomAR (P2P all-reduce)。
# PP 生产路径下 TP=1 → is_tp_enabled()=False → all_reduce 返回 clone, all_gather 返回自身。
#
# 源实现文件: engine/tp_layers/distributed.py
import os

import torch
import torch.distributed as dist

try:
    from vllm import _custom_ops as ops  # allocate_shared_buffer_and_handle / init_custom_ar ...
    _HAS_VLLM_OPS = True
except Exception:  # pragma: no cover
    ops = None
    _HAS_VLLM_OPS = False


_TP_GROUP = None


class TPGroup:
    """TP 进程组句柄。PP 生产路径下 tp_size=1 (is_tp_enabled()=False)。"""

    def __init__(self, rank: int = 0, size: int = 1, group=None):
        self.rank = rank
        self.size = size
        self.group = group

    def is_tp_enabled(self) -> bool:
        return self.size > 1 and (dist.is_initialized() or self.group is not None)


def init_tp_process_group() -> TPGroup:
    """初始化 TP 进程组。dist 未初始化 → tp_size=1 直通 (PP 生产路径)。幂等。

    Round-2 ⑦ 修复: TP 大小显式由 VLLM_TENSOR_PARALLEL_SIZE 决定 (默认 1),
    不再把 world_size 当作 TP size。Round-1 在 PP=2 (world_size=2) 下误判
    tp_size=2 → is_tp_enabled()=True → 每层 all_reduce 错误激活 (PP 边界是
    单向 P2P, 无集合通信), 且与 PP P2P 语义冲突。
    判定: tp_size=1 → 直通 (size=1, 永不 enable); tp_size==world_size → 全组即 TP;
    其余组合 (TP 子组 < world, 即 TP×PP 混合) 暂不支持 → RuntimeError。"""
    global _TP_GROUP
    if _TP_GROUP is not None:
        return _TP_GROUP
    tp_size = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    if dist.is_initialized():
        rank = dist.get_rank()
        world = dist.get_world_size()
        if tp_size <= 1:
            # PP 生产路径: TP 轨 no-op, is_tp_enabled()=False
            _TP_GROUP = TPGroup(rank=0, size=1, group=None)
        elif tp_size == world:
            _TP_GROUP = TPGroup(rank=rank, size=tp_size, group=None)
        elif tp_size > world:
            raise RuntimeError(
                f"VLLM_TENSOR_PARALLEL_SIZE={tp_size} > world_size={world} — 非法 TP 配置")
        else:
            raise RuntimeError(
                f"TP 子组 ({tp_size}) < world_size ({world}) 的 TP×PP 混合暂不支持 — "
                f"当前生产路径 PP=2 保持 tp_size=1 直通")
    else:
        _TP_GROUP = TPGroup(rank=0, size=1, group=None)
    return _TP_GROUP


def get_tp_group() -> TPGroup:
    """获取 TP 进程组 (未 init 自动 init 为 tp_size=1 直通)。"""
    global _TP_GROUP
    if _TP_GROUP is None:
        init_tp_process_group()
    return _TP_GROUP


def is_tp_enabled() -> bool:
    """TP 是否启用 (tp_size>1 且 dist 已初始化)。PP 生产路径 = False。"""
    return get_tp_group().is_tp_enabled()


# ---------------------------------------------------------------------------
# all_reduce_sum
# ---------------------------------------------------------------------------
_custom_ar_handle = None


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """跨 TP ranks 求和, 不改变 shape。TP=1 → return x.clone() (custom_op 禁别名)。"""
    if not is_tp_enabled():
        return x.clone()
    global _custom_ar_handle
    if _custom_ar_handle is not None and _HAS_VLLM_OPS:
        try:
            return _custom_ar_handle.all_reduce(x, registered=False)
        except Exception:
            pass
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def all_gather_last_dim(x: torch.Tensor) -> torch.Tensor:
    """沿最后一维拼接各 rank 分片。TP=1 → 返回 x 自身。"""
    tp = get_tp_group()
    if not tp.is_tp_enabled():
        return x
    out_list = [torch.empty_like(x) for _ in range(tp.size)]
    dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=-1)


# ---------------------------------------------------------------------------
# CustomAR (P2P All-Reduce) — 可选优化, 失败自动 fallback NCCL
# ---------------------------------------------------------------------------
def init_custom_ar() -> None:
    """初始化 CustomAR P2P all-reduce。⚠️ 整个 init 必须在 try/except。
    失败 → _custom_ar_handle=None 触发 NCCL fallback。world_size=1 立即 return (no-op)。"""
    global _custom_ar_handle
    tp = get_tp_group()
    if not tp.is_tp_enabled() or not _HAS_VLLM_OPS:
        _custom_ar_handle = None
        return
    try:
        dist.barrier()
        gloo_group = dist.new_group(backend="gloo")
        max_size = 16 * 1024 * 1024  # 16 MB
        raw_ptr, ipc_handle = ops.allocate_shared_buffer_and_handle(max_size)
        world_size = tp.size
        rank = tp.rank
        handles = [None] * world_size
        dist.all_gather_object(handles, ipc_handle, group=gloo_group)
        pointers = [ops.open_mem_handle(h) if i != rank else raw_ptr
                    for i, h in enumerate(handles)]
        rank_data = ops.meta_size()
        _custom_ar_handle = ops.init_custom_ar(pointers, rank_data, rank, True)
        ops.register_buffer(_custom_ar_handle, pointers)
        dist.barrier()
    except Exception:
        _custom_ar_handle = None  # 确保 None → NCCL fallback


def register_graph_buffers():
    """CUDA Graph 路径的 buffer 注册 (eager nocompile 不用)。"""
    return None
