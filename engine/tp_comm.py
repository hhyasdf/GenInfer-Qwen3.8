# Phase 2 — TP 通信别名模块 (re-export from engine/tp_layers/distributed.py)。
from .tp_layers.distributed import (
    init_tp_process_group, get_tp_group, is_tp_enabled,
    all_reduce_sum, all_gather_last_dim, init_custom_ar,
)
