# Phase 14 — PP 调度气泡 (pp_scheduler_contracts.md)。
#
# 核心语义:
#   - 1F1B 气泡下界 b ≈ (d-1)/m, d=stage 数, m=micro-batches。
#   - micro-batch / chunked prefill 对齐: 把长 prefill 切 chunk, 按 microbatch 旋转。
#   - ⛔ GDN 层不可 ubatch: chunked prefill 必须按序列边界对齐。
#   - vLLM v1 调度器在 PP rank 间复制 (同一调度决策, lockstep)。
#
# Round-2 D-5 修复:
#   - align_chunked_prefill 无 seq_lengths + sequence_boundary=True → 均匀网格吸附:
#     unit = 最大 2 的幂 ≤ chunk_size 且整除 num_tokens, 块大小 = (chunk//unit)*unit。
#     (512,200)→[128,128,128,128]; (1024,256)→[256,256,256,256]。Round-1 该路径朴素
#     切分 → [200,200,112] 切开序列 (PP-SCHED-004 FAIL)。
#   - sequence_boundary=False → 朴素切分 (不吸附, 负向对照)。
#   - schedule(requests) 返回请求级 list[list[int]]: 每个 microbatch 含完整请求,
#     不拆分/不丢弃, 空批合法 (continuous batching)。
#
# 参考: SGLang 1F1B event_loop_pp (scheduler_pp_mixin.py), vLLM ubatch_size/enable_dbo。
from __future__ import annotations


class PPSchedulerError(ValueError):
    """调度违规统一异常。消息含陷阱编号 (PP-SCHED-00X)。"""


def estimate_pipeline_bubble(depth: int, micro_batches: int) -> float:
    """1F1B 气泡下界 = (d-1)/m。d=stage 数, m=micro-batches。"""
    if micro_batches <= 0:
        raise PPSchedulerError("PP-SCHED-003: micro_batches 必须 > 0")
    if depth <= 1:
        return 0.0
    return (depth - 1) / micro_batches


def align_chunked_prefill(num_tokens: int, chunk_size: int,
                          sequence_boundary: bool = True,
                          gdn_layer: bool = False,
                          seq_lengths: list[int] | None = None) -> list[int]:
    """返回各 microbatch 的 token 数。

    sequence_boundary=True 或 gdn_layer=True 时, chunk 边界不得落在序列中间 —
    吸附到最近序列边界。seq_lengths 提供本 batch 逐序列 token 长度 (贪心整序列
    打包)。无 seq_lengths 且 sequence_boundary=True → 均匀网格吸附
    (unit = 最大 2 的幂 ≤ chunk 且整除 num_tokens)。
    sequence_boundary=False → 朴素切分 (不吸附)。
    """
    if chunk_size <= 0:
        raise PPSchedulerError("PP-SCHED-004: chunk_size 必须 > 0")
    if num_tokens <= 0:
        return []
    # 朴素切分 (sequence_boundary=False): 不吸附, 允许切开序列
    if not sequence_boundary:
        chunks = []
        remaining = num_tokens
        while remaining > 0:
            take = min(chunk_size, remaining)
            chunks.append(take)
            remaining -= take
        return chunks
    # 按序列边界对齐: seq_lengths 给出每序列长度, 贪心把整序列打包进 chunk (不切序列)
    if seq_lengths:
        chunks = []
        cur = 0
        for slen in seq_lengths:
            if cur + slen <= chunk_size:
                cur += slen
            else:
                if cur > 0:
                    chunks.append(cur)
                # 单序列超 chunk_size → 整序列独占一个 chunk (GDN 不切序列)
                cur = slen
        if cur > 0:
            chunks.append(cur)
        return chunks
    # 无 seq_lengths + sequence_boundary=True: 均匀网格吸附 —
    # unit = 最大 2 的幂 ≤ chunk_size 且整除 num_tokens; 块大小 = (chunk//unit)*unit,
    # 每块均为 unit 整数倍 → 边界必落在序列边界上。
    # (512,200)→unit=128→[128,128,128,128]; (1024,256)→unit=256→[256,256,256,256]
    unit = 1
    p = 1
    while p * 2 <= chunk_size:
        p *= 2
    while p > 1 and num_tokens % p != 0:
        p //= 2
    unit = p
    eff = (chunk_size // unit) * unit
    chunks = []
    remaining = num_tokens
    while remaining > eff:
        chunks.append(eff)
        remaining -= eff
    if remaining > 0:
        chunks.append(remaining)
    return chunks


class PPScheduler:
    """流水线调度器 (1F1B microbatch 分配)。"""

    def __init__(self, depth: int, micro_batches: int):
        self.depth = depth
        self.micro_batches = micro_batches

    def schedule(self, requests) -> list[list[int]]:
        """1F1B microbatch 分配 — 请求级 list[list[int]]。
        requests: 每个元素是 token 数 (int) 或含 sequence_length/num_tokens 的 dict。
        返回 microbatch 列表, 每个 microbatch 含**完整请求**的 token 数:
        不拆分单请求, 不丢弃请求, 空批 (无请求) 合法。continuous batching —
        单条请求即被接纳, 不要求凑满固定 batch。"""
        seq_lengths = [int(r.get("sequence_length", r.get("num_tokens", 1)))
                       if isinstance(r, dict) else int(r) for r in requests]
        if not seq_lengths:
            return []
        mbs: list[list[int]] = []
        cur: list[int] = []
        for slen in seq_lengths:
            cur.append(slen)
            if len(cur) >= self.micro_batches:
                mbs.append(cur)
                cur = []
        if cur:
            mbs.append(cur)
        return mbs

    def bubble_fraction(self) -> float:
        """实际气泡 (须 ≥ 下界且逼近之)。1F1B 理想下气泡 = (d-1)/m。"""
        if self.micro_batches <= 0:
            return 1.0
        if self.depth <= 1:
            return 0.0
        # 1F1B 调度理想气泡逼近下界
        return (self.depth - 1) / self.micro_batches

    def validate_no_gdn_ubatch(self, stage_has_gdn: bool, ubatch_size: int) -> None:
        """⛔ stage 含 GDN 且 ubatch_size>1 → 抛 PPSchedulerError(PP-SCHED-002)。"""
        if stage_has_gdn and ubatch_size > 1:
            raise PPSchedulerError(
                "PP-SCHED-002: stage 含 GDN 层禁 ubatch — recurrent state 不能跨 microbatch 切")
