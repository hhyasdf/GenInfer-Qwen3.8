# Phase 8 — Runner (framework_contracts.md)。
# runner.run(batch, is_prefill, temperature) → output{next_tokens}; runner.get_num_free_blocks()。
# 持有 paged KV block pool (block_size=256, MAX_BLOCKS=160) + GDN recurrent state。
import torch
from .sequence import Sequence
from .sampler import Sampler


class ModelRunner:
    """模型执行器: prefill / decode 前向 + 采样。持有 KV block pool 与 GDN state。"""

    def __init__(self, model, scheduler, block_size=256, max_blocks=160, device="cpu"):
        self.model = model
        self.scheduler = scheduler
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.device = device
        self.sampler = Sampler()
        # paged KV pool: [max_blocks, block_size, ...] — 简化为按 layer 的 token 级 cache
        # (per-stage FA 层 KV 按层累积; GDN state 为每层常数 recurrent state)
        self._kv_cache = {}     # {layer_idx: {"k":..., "v":...}}
        self._gdn_states = {}   # {layer_idx: state}
        self._free_blocks = max_blocks
        self.model.to(device)
        self.model.eval()

    def get_num_free_blocks(self) -> int:
        return self._free_blocks

    def _reset_cache(self):
        self._kv_cache = {}
        self._gdn_states = {}
        self._free_blocks = self.max_blocks

    @torch.no_grad()
    def run(self, batch: list, is_prefill: bool, temperature: float = 0.0) -> dict:
        """执行一个 batch (prefill 或 decode)。返回 {'next_tokens': {req_id: token}}。"""
        if is_prefill:
            self._reset_cache()
        # 本变体单序列执行 (LLM_CONCURRENCY=1); 取第一个请求
        seq = batch[0]
        if is_prefill:
            input_ids = torch.tensor([seq.input_ids], dtype=torch.long, device=self.device)
        else:
            # decode: 只喂最后一个 token
            last = seq.output_ids[-1] if seq.output_ids else seq.input_ids[-1]
            input_ids = torch.tensor([[last]], dtype=torch.long, device=self.device)
        # position_ids: 本变体用自增 (无 mrope 输入)
        pos = seq.kv_len - 1
        position_ids = torch.stack([torch.tensor([pos])] * 3)
        out = self.model.forward(
            input_ids=input_ids, kv_cache=self._kv_cache,
            gdn_states=self._gdn_states, position_ids=position_ids)
        if out[0] is not None:
            logits = out[0]
            self._kv_cache, self._gdn_states = out[1], out[2]
            sp = dict(seq.sampling_params)
            sp["temperature"] = temperature
            toks = self.sampler.sample(logits, sp)
            return {"next_tokens": {seq.request_id: toks[0]}}
        # 非末 stage: 无 logits
        return {"next_tokens": {}}
