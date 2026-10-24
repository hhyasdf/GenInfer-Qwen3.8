# Phase 8 — Scheduler (framework_contracts.md)。
# WAITING→RUNNING_PREFILL→RUNNING_DECODE→FINISHED (+REJECTED)。无 preempt()。
# block_size=256; MAX_BLOCKS=160。
from .sequence import Sequence, SequenceStatus

BLOCK_SIZE = 256
MAX_BLOCKS = 160


class Scheduler:
    """请求调度器。schedule(waiting, running, num_free) → result{is_prefill, batch}。
    postprocess(batch, success, tokens) 推进状态机。"""

    def __init__(self, block_size: int = BLOCK_SIZE, max_blocks: int = MAX_BLOCKS):
        self.block_size = block_size
        self.max_blocks = max_blocks

    def _need_blocks(self, seq: Sequence, is_prefill: bool) -> int:
        # prefill: 整段 input_ids; decode: 已生成+1
        toks = seq.num_prompt_tokens if is_prefill else seq.kv_len + 1
        return -(-toks // self.block_size)  # ceil

    def schedule(self, waiting: list, running: list, num_free: int) -> dict:
        """优先 prefill 一个 WAITING 请求; 否则 decode RUNNING 请求。
        返回 {'is_prefill': bool, 'batch': [Sequence], 'num_tokens': int}。"""
        # 1) prefill: 取第一个能放进 num_free 的 WAITING
        for seq in waiting:
            need = self._need_blocks(seq, is_prefill=True)
            if need <= num_free:
                seq.status = SequenceStatus.RUNNING_PREFILL
                return {"is_prefill": True, "batch": [seq],
                        "num_tokens": seq.num_prompt_tokens}
        # 2) decode: RUNNING 请求
        if running:
            batch = list(running)
            for seq in batch:
                if seq.status == SequenceStatus.RUNNING_PREFILL:
                    seq.status = SequenceStatus.RUNNING_DECODE
            return {"is_prefill": False, "batch": batch,
                    "num_tokens": sum(s.kv_len for s in batch)}
        # 3) 无请求
        return {"is_prefill": False, "batch": [], "num_tokens": 0}

    def postprocess(self, batch: list, success: bool, tokens=None):
        """一步后处理: decode 成功则 append token; 达 max_tokens 则 FINISHED。"""
        if tokens is None:
            tokens = {}
        for seq in batch:
            if not success:
                seq.status = SequenceStatus.REJECTED
                continue
            tok = tokens.get(seq.request_id)
            if tok is not None:
                seq.append_token(tok)
            # 完成判定
            if seq.status == SequenceStatus.RUNNING_PREFILL:
                seq.status = SequenceStatus.RUNNING_DECODE
            if len(seq.output_ids) >= seq.sampling_params.get("max_tokens", 64):
                seq.status = SequenceStatus.FINISHED

    def get_num_free_blocks(self) -> int:
        """当前空闲 block 数 (由 runner 的 paged cache 实际决定, 此返回上限)。"""
        return self.max_blocks
