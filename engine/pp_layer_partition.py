# Phase 13 — PP 层分区 (pp_layer_partition_contracts.md)。
#
# 核心语义 (唯一契约来源):
#   - 按 (层号-1) % full_attention_interval == 3 判定 FA 层 (GDN 层不满足)。
#     区间末尾必须落在 FA 层 —— 这样 KV cache 才能按区间整段分配,
#     不会把 FA 层拆到两个 stage。
#   - 每 stage 的层数必须整除 full_attention_interval (区间内 FA 层等距)。
#   - 分区结果 = list[list[int]] —— 每 stage 的**层号索引列表**
#     (如 [[0..27],[28..63]]), 不是大小 [28,36]。Round-1 误返回大小并被
#     llm_engine 当作层号切片 (partition[rank]) → 语义错乱 (D-3 根因)。
#   - MTP 块 (独立 blk.64) 按"与哪段物理连续"原则分配给末 stage 末段所在卡。
#   - get_pp_indices 兼容两种入参: 索引列表 (compute_pp_layer_partition 产物)
#     与大小列表 (向后兼容); 不合法大小 (len≠pp_size / sum≠total) 必抛
#     PPLayerPartitionError, 不静默截断。
#
# Round-2 D-3 修复:
#   - 返回 list[list[int]] 层号索引 (非大小); vram_ratio 公式修正为
#     last_stage_size = round(num_layers * r / (1 + r)) —— Round-1 用
#     num_layers/(1+r) 把 stage0 算成了 32 而非 28, 且未对齐 FA 边界。
#   - vram_ratio 语义: r = 末段/首段 显存比。r=1.25 (22.5/16 GB 异构双卡)
#     → stage1 = round(64×1.25/2.25)=36 (FA 对齐 36%4==0), stage0=28。
#   - get_pp_indices 对索引列表校验连续覆盖 (拼接 == range(total)),
#     对大小列表校验 len+sum, 违例抛 PPLayerPartitionError。
from __future__ import annotations


class PPLayerPartitionError(ValueError):
    """分区不合法: FA 边界 / 整除性 / MTP 归属。消息含陷阱编号 (PP-PART-00X)。"""


def _snap_fa_boundary(size: int, full_attention_interval: int) -> int:
    """把 stage 层数下取到 full_attention_interval 的整数倍 (区间末尾落在 FA 层)。"""
    return (size // full_attention_interval) * full_attention_interval


def compute_pp_layer_partition(num_layers: int, pp_size: int,
                               full_attention_interval: int = 4,
                               vram_ratio: float = 1.0) -> list[list[int]]:
    """返回每 stage 的**层号索引列表** list[list[int]] (如 [[0..27],[28..63]]),
    区间末尾落在 FA 层 (每段长度是 interval 整数倍)。

    vram_ratio=r: 末段层数 = round(num_layers * r / (1+r)), 再下取到 FA 对齐。
      r=1.0 → 均分; r=1.25 → 64 层分 [0..27]/[28..63] (异构双卡 16/22.5 GB)。
    pp_size==1 → [list(range(num_layers))]。
    违例 (FA 边界不满足 / 层数不整除) → PPLayerPartitionError (PP-PART-003)。
    """
    if num_layers <= 0 or pp_size <= 0:
        raise PPLayerPartitionError(
            f"PP-PART-003: invalid num_layers={num_layers} pp_size={pp_size}")
    if pp_size == 1:
        return [list(range(num_layers))]
    # 首段 = 均分基底, 末段按 vram_ratio 加权 (显存大的卡分更多层)
    base = num_layers // pp_size
    last_size = round(num_layers * vram_ratio / (1.0 + vram_ratio)) if pp_size == 2 \
        else base
    # 下取到 FA 对齐 (区间末尾落在 FA 层)
    last_aligned = _snap_fa_boundary(last_size, full_attention_interval)
    if last_aligned < full_attention_interval:
        raise PPLayerPartitionError(
            f"PP-PART-003: last stage size {last_aligned} < FA interval "
            f"{full_attention_interval} — cannot hold a complete attention block")
    first_size = num_layers - last_aligned
    if first_size % full_attention_interval != 0:
        # 首段也必须整除 (区间内 FA 层等距)
        first_size = _snap_fa_boundary(first_size, full_attention_interval)
        last_aligned = num_layers - first_size
        if last_aligned % full_attention_interval != 0:
            raise PPLayerPartitionError(
                f"PP-PART-003: cannot split {num_layers} layers into "
                f"{pp_size} FA-aligned stages (interval={full_attention_interval})")
    # R3: 返回 list[list[int]] 层号索引 (对照不可变门禁 test_phase13: partition[0] ==
    # list(range(0,28)), partition[1] == list(range(28,64)))。Round-2 误返回 sizes
    # [28,36] → 门禁 :51 [l for stage in partition for l in stage] TypeError (FAIL-1)。
    # get_pp_indices / get_stage_layers / assign_mtp_device 的索引列表分支 (:107-119)
    # 校验连续覆盖拼接 == range(total), 与门禁一致。
    def _to_indices(sizes: list[int]) -> list[list[int]]:
        part: list[list[int]] = []
        start = 0
        for s in sizes:
            part.append(list(range(start, start + s)))
            start += s
        return part
    if pp_size == 2:
        return _to_indices([first_size, last_aligned])
    base = _snap_fa_boundary(num_layers // pp_size, full_attention_interval)
    sizes = [base] * (pp_size - 1)
    sizes.append(num_layers - base * (pp_size - 1))
    return _to_indices(sizes)


def get_stage_layers(num_layers: int, pp_rank: int, pp_size: int,
                     partition_list: list | None = None) -> list[int]:
    """返回 pp_rank stage 负责的层号列表。partition_list 缺省 → 均分。"""
    start, end = get_pp_indices(num_layers, pp_rank, pp_size, partition_list)
    return list(range(start, end))


def get_pp_indices(num_hidden_layers: int, pp_rank: int, pp_size: int,
                   partition_list: list | None = None) -> tuple[int, int]:
    """(start, end) 闭开区间, 本 stage 负责 blk[start..end-1]。
    partition_list 支持两种形态:
      - 索引列表 list[list[int]] (compute_pp_layer_partition 产物):
        校验连续覆盖拼接 == range(total), 违例抛 PPLayerPartitionError。
      - 大小列表 list[int] (向后兼容): len==pp_size 且 sum==total,
        违例抛 PPLayerPartitionError (PP-PART-002), 不静默截断。
    缺省 → 均分 (余数给前 rem 个 stage)。
    """
    if partition_list is None:
        base = num_hidden_layers // pp_size
        rem = num_hidden_layers % pp_size
        size = base + (1 if pp_rank < rem else 0)
        start = pp_rank * base + min(pp_rank, rem)
        return start, start + size
    if not (0 <= pp_rank < pp_size):
        raise PPLayerPartitionError(
            f"PP-PART-002: pp_rank={pp_rank} out of range for pp_size={pp_size}")
    # 形态判定: 第一个元素是 list → 索引列表; 是 int → 大小列表
    if partition_list and isinstance(partition_list[0], (list, range)):
        if len(partition_list) != pp_size:
            raise PPLayerPartitionError(
                f"PP-PART-002: partition_list has {len(partition_list)} stages "
                f"but pp_size={pp_size}")
        # 校验连续覆盖: 拼接后 == range(total)
        flat = [il for seg in partition_list for il in seg]
        if flat != list(range(num_hidden_layers)):
            raise PPLayerPartitionError(
                f"PP-PART-002: index partition does not cover 0..{num_hidden_layers - 1} "
                f"contiguously (got {len(flat)} layers)")
        seg = list(partition_list[pp_rank])
        return seg[0], seg[-1] + 1
    # 大小列表
    if len(partition_list) != pp_size:
        raise PPLayerPartitionError(
            f"PP-PART-002: partition_list has {len(partition_list)} stages "
            f"but pp_size={pp_size}")
    if sum(partition_list) != num_hidden_layers:
        raise PPLayerPartitionError(
            f"PP-PART-002: partition sizes sum to {sum(partition_list)} "
            f"!= num_hidden_layers={num_hidden_layers}")
    start = sum(partition_list[:pp_rank])
    return start, start + partition_list[pp_rank]


def is_full_attention(il: int, full_attention_interval: int = 4) -> bool:
    """il % interval == interval-1 → FA 层 (对照 config.py fa_layers / qwen.py decoder)。
    例: interval=4 → FA 在 il=3,7,11,...,63; GDN 在其余。"""
    return il % full_attention_interval == full_attention_interval - 1


# self_check (L0.5) 契约名 — 同义别名
is_fa_layer = is_full_attention


def assign_mtp_device(partition: list, mtp_layer: int, pp_size: int) -> int:
    """MTP 块 (独立 blk.64) 归属哪个 stage —— 按"与末段物理连续"原则给末段所在 stage。
    partition 支持索引列表或大小列表。mtp_layer 落在某 stage 区间内 → 该 stage。
    否则 → 末 stage。"""
    if partition and isinstance(partition[0], (list, range)):
        for i, seg in enumerate(partition):
            if seg[0] <= mtp_layer < seg[-1] + 1:
                return i
        return pp_size - 1
    start = 0
    for i, size in enumerate(partition):
        if start <= mtp_layer < start + size:
            return i
        start += size
    return pp_size - 1


def get_mtp_block(partition: list, mtp_layer: int) -> list[str]:
    """返回 MTP 块归属 stage 及其实际 GGUF 张量名清单 (15 个)。
    qwen35 MTP blk.64 实际布局 (GGUF dump 确认):
      FA 注意力独立张量 attn_q/k/v/attn_output (非融合 qkv),
      attn_q_norm/attn_k_norm (F32), norm 命名 attn_norm/post_attention_norm,
      MLP 独立 ffn_gate/ffn_up (非融合 ffn_gate_up), ffn_down,
      nextn.eh_proj/enorm/hnorm/shared_head_norm (无 embed_tokens, 无 shared_head_head)。
    """
    stage = assign_mtp_device(partition, mtp_layer, len(partition))
    names = [
        f"blk.{mtp_layer}.attn_norm.weight",
        f"blk.{mtp_layer}.post_attention_norm.weight",
        f"blk.{mtp_layer}.attn_q.weight",
        f"blk.{mtp_layer}.attn_k.weight",
        f"blk.{mtp_layer}.attn_v.weight",
        f"blk.{mtp_layer}.attn_q_norm.weight",
        f"blk.{mtp_layer}.attn_k_norm.weight",
        f"blk.{mtp_layer}.attn_output.weight",
        f"blk.{mtp_layer}.ffn_gate.weight",
        f"blk.{mtp_layer}.ffn_up.weight",
        f"blk.{mtp_layer}.ffn_down.weight",
        f"blk.{mtp_layer}.nextn.eh_proj.weight",
        f"blk.{mtp_layer}.nextn.enorm.weight",
        f"blk.{mtp_layer}.nextn.hnorm.weight",
        f"blk.{mtp_layer}.nextn.shared_head_norm.weight",
    ]
    return names
