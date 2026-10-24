# Phase 4 — TP Embedding & LM Head (tp_embedding_contracts.md)。
#
# VocabParallelEmbedding: mask → masked_fill → all_reduce_sum (非 all_gather)。
# ParallelLMHead: F.linear → all_gather_last_dim。
# PP 生产路径下 TP=1: VocabParallelEmbedding 退化为 F.embedding, ParallelLMHead 退化为 F.linear。
# 源实现文件: engine/tp_layers/embedding.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .distributed import get_tp_group, is_tp_enabled, all_reduce_sum, all_gather_last_dim


class VocabParallelEmbedding(nn.Module):
    """input_ids [B,T] int64 → [B,T,hidden] (after all_reduce_sum)。
    local_weight [vocab/tp, hidden]。超本地词表 token → mask 置 0 → all_reduce_sum。"""

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        tp = get_tp_group()
        self.tp_size = tp.size
        self.tp_rank = tp.rank
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.local_vocab_size = vocab_size if not is_tp_enabled() else vocab_size // tp.size
        self.vocab_start = tp.rank * (vocab_size // tp.size) if is_tp_enabled() else 0
        self.vocab_end = self.vocab_start + self.local_vocab_size
        self.weight = nn.Parameter(torch.empty(self.local_vocab_size, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        local_ids = (input_ids - self.vocab_start).masked_fill(~mask, 0)
        out = F.embedding(local_ids, self.weight)  # [B,T,hidden]
        out = out.masked_fill((~mask).unsqueeze(-1), 0)
        return all_reduce_sum(out)  # EMBED-001: all_reduce_sum, 非 all_gather

    def load_weight_shard(self, weight_tensor):
        if weight_tensor.shape == self.weight.shape:
            self.weight.copy_(weight_tensor)
        else:
            self.weight.copy_(weight_tensor[self.vocab_start:self.vocab_end])


class ParallelLMHead(nn.Module):
    """input_hidden [B,T,hidden] → logits [B,T,vocab] (after all_gather_last_dim)。
    local_weight [vocab/tp, hidden]。"""

    def __init__(self, vocab_size, hidden_size, gather_output=True):
        super().__init__()
        tp = get_tp_group()
        self.tp_size = tp.size
        self.tp_rank = tp.rank
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.local_vocab_size = vocab_size if not is_tp_enabled() else vocab_size // tp.size
        self.gather_output = gather_output
        self.weight = nn.Parameter(torch.empty(self.local_vocab_size, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        local_logits = F.linear(x, self.weight)  # [B,T,vocab/tp]
        if self.gather_output and is_tp_enabled():
            return all_gather_last_dim(local_logits)  # [B,T,vocab]
        return local_logits

    def load_weight_shard(self, weight_tensor):
        if weight_tensor.shape == self.weight.shape:
            self.weight.copy_(weight_tensor)
        else:
            self.weight.copy_(weight_tensor[:self.local_vocab_size])
