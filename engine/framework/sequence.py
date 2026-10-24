# Phase 8 — Sequence + SequenceStatus (framework_contracts.md)。
# engine.framework.sequence 必导出: Sequence, SequenceStatus。
from enum import Enum, auto


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING_PREFILL = auto()
    RUNNING_DECODE = auto()
    FINISHED = auto()
    REJECTED = auto()


class Sequence:
    """请求序列。seq.sampling_params = {'max_tokens','temperature','top_p'}。"""

    def __init__(self, request_id: str, input_ids, output_ids=None,
                 status: SequenceStatus = SequenceStatus.WAITING):
        self.request_id = request_id
        self.input_ids = list(input_ids)
        self.output_ids = list(output_ids) if output_ids is not None else []
        self.status = status
        self.sampling_params = {"max_tokens": 64, "temperature": 0.0, "top_p": 1.0}
        self.num_prompt_tokens = len(self.input_ids)

    @property
    def kv_len(self) -> int:
        """当前已占 KV token 数 = prompt + 已生成。"""
        return len(self.input_ids) + len(self.output_ids)

    @property
    def total_len(self) -> int:
        return self.kv_len

    def append_token(self, token_id: int):
        self.output_ids.append(token_id)

    def is_finished(self) -> bool:
        return self.status in (SequenceStatus.FINISHED, SequenceStatus.REJECTED)

    def __repr__(self):
        return (f"Sequence({self.request_id}, status={self.status.name}, "
                f"in={len(self.input_ids)}, out={len(self.output_ids)})")
