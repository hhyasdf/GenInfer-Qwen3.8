# Phase 3 — TP 线性层 (tp_linear_contracts.md)。
#
# 4 种: ColumnParallelLinear, RowParallelLinear, QKVColumnParallelLinear, MergedColumnParallelLinear。
# PP 生产路径下 TP=1: 各层退化为普通 nn.Linear (无切片, 无集合通信)。
# 源实现文件: engine/tp_layers/linear.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .distributed import get_tp_group, is_tp_enabled, all_reduce_sum, all_gather_last_dim


class ColumnParallelLinear(nn.Module):
    """weight [out/tp, in]。输入 [B,T,in]。输出 [B,T,out/tp] (no gather) 或 [B,T,out] (gather)。"""

    def __init__(self, hidden_size, output_size_per_partition, bias=False, gather_output=True):
        super().__init__()
        tp = get_tp_group()
        self.tp_size = tp.size
        self.tp_rank = tp.rank
        self.hidden_size = hidden_size
        self.output_size_per_partition = output_size_per_partition
        self.gather_output = gather_output
        out = output_size_per_partition if not is_tp_enabled() else output_size_per_partition
        self.weight = nn.Parameter(torch.empty(output_size_per_partition, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size_per_partition))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)  # [B,T,out/tp]
        if self.gather_output and is_tp_enabled():
            y = all_gather_last_dim(y)
        return y

    def load_weight_shard(self, weight_tensor):
        if weight_tensor.shape == self.weight.shape:
            self.weight.copy_(weight_tensor)
        else:
            chunk = weight_tensor.shape[0] // self.tp_size
            self.weight.copy_(weight_tensor[self.tp_rank * chunk:(self.tp_rank + 1) * chunk])


class RowParallelLinear(nn.Module):
    """weight [out, in/tp]。输入 [B,T,in/tp]。partial [B,T,out] → all_reduce 后 [B,T,out]。"""

    def __init__(self, input_size, output_size, bias=False):
        super().__init__()
        tp = get_tp_group()
        self.tp_size = tp.size
        self.tp_rank = tp.rank
        self.input_size = input_size
        self.output_size = output_size
        in_local = input_size if not is_tp_enabled() else input_size
        self.weight = nn.Parameter(torch.empty(output_size, in_local))
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        y = F.linear(x, self.weight)  # partial [B,T,out]
        if is_tp_enabled():
            y = all_reduce_sum(y)
        else:
            y = y + self.bias if self.bias is not None else y
        return y

    def load_weight_shard(self, weight_tensor):
        if weight_tensor.shape == self.weight.shape:
            self.weight.copy_(weight_tensor)
        else:
            chunk = weight_tensor.shape[1] // self.tp_size
            self.weight.copy_(weight_tensor[:, self.tp_rank * chunk:(self.tp_rank + 1) * chunk])


class QKVColumnParallelLinear(nn.Module):
    """weight per rank [q_size + 2*kv_size, hidden] (前 Q 中 K 后 V)。
    head_dim = cfg.head_dim。q_size = num_heads*head_dim//tp。kv_size = kv_heads_local*head_dim。"""

    def __init__(self, hidden_size, head_dim, total_num_heads, total_num_kv_heads,
                 bias=False, gather_output=False):
        super().__init__()
        tp = get_tp_group()
        self.tp_size = tp.size
        self.tp_rank = tp.rank
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = total_num_heads if not is_tp_enabled() else total_num_heads // tp.size
        if total_num_kv_heads >= tp.size:
            self.num_kv_heads = total_num_kv_heads if not is_tp_enabled() else total_num_kv_heads // tp.size
            self.kv_head_replica = 1
        else:
            self.num_kv_heads = 1
            self.kv_head_replica = tp.size // total_num_kv_heads
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        out = self.q_size + 2 * self.kv_size
        self.weight = nn.Parameter(torch.empty(out, hidden_size))
        self.bias = nn.Parameter(torch.zeros(out)) if bias else None
        self.gather_output = gather_output
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)  # [B,T,q_size+2*kv_size]
        if self.gather_output and is_tp_enabled():
            y = all_gather_last_dim(y)
        q, k, v = y.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v

    def load_weight_shard(self, weight_tensor):
        if weight_tensor.shape == self.weight.shape:
            self.weight.copy_(weight_tensor)
        else:
            chunk = weight_tensor.shape[0] // self.tp_size
            self.weight.copy_(weight_tensor[self.tp_rank * chunk:(self.tp_rank + 1) * chunk])


class MergedColumnParallelLinear(nn.Module):
    """gate_up weight per rank [2*intermediate/tp, hidden]。输出 [B,T,2*inter/tp] (前 gate 后 up)。"""

    def __init__(self, hidden_size, intermediate_size, bias=False, gather_output=False):
        super().__init__()
        tp = get_tp_group()
        self.tp_size = tp.size
        self.tp_rank = tp.rank
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size if not is_tp_enabled() else intermediate_size // tp.size
        out = 2 * self.intermediate_size
        self.weight = nn.Parameter(torch.empty(out, hidden_size))
        self.bias = nn.Parameter(torch.zeros(out)) if bias else None
        self.gather_output = gather_output
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)  # [B,T,2*inter/tp]
        if self.gather_output and is_tp_enabled():
            y = all_gather_last_dim(y)
        return y

    def load_weight_shard(self, weight_tensor):
        if weight_tensor.shape == self.weight.shape:
            self.weight.copy_(weight_tensor)
        else:
            chunk = weight_tensor.shape[0] // self.tp_size
            self.weight.copy_(weight_tensor[self.tp_rank * chunk:(self.tp_rank + 1) * chunk])
