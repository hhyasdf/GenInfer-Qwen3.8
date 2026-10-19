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


# ===========================================================================
# Round-4 fix ② — 完整 4 维 VRAM 预算切层公式 (代码显式, self_check k3 可重算)。
#
# R3 根因: vram_ratio 只按权重大小切层 (stage0 10.16GiB / stage1 13.13GiB), 未给
# dequant 临时 + 激活 + KV + GDN state 留预算 → stage0 的 16GB 卡在 prefill 最大
# FFN GEMM 的旧 dequant (f32 out + int64 ql/qh, ~4.5GB 临时) 触顶 → 0 token。
# R4 fix ① 低温 dequant 后 dequant 临时 = 每-GEMM bf16 权重 (stage0 峰值 FFN 170MB)。
#
# 每 stage 预算 = W (权重, 精确账本) + D (dequant 临时峰) + A (激活峰)
#                + K (KV cache 配额, q8_0 2176B/token/FA 层) + G (GDN 循环状态)。
# 数据来源: gguf_fields_values.txt 维度 + gguf_dump n_bytes 逐张量精确账本
#           (exploration_report.md R2 ⑥ / R3 方向 ①) + AGGREGATE_REPORT_R3.md
#           VRAM 采样 (R3 GPU0 峰值 15975MiB 死因)。
# ===========================================================================
Q6_K_BPP = 210 / 256              # Q6_K 0.8203125 B/elem (210B/256elem 超级块)
Q8_0_BPP = 34 / 32                # Q8_0 1.0625 B/elem (34B/32elem 块)
KV_BPE_Q8_0 = 1 + 1 / 32          # llama-server --cache-type-k/v q8_0: 1.0625 B/elem
KV_BYTES_PER_TOKEN_FA = 2 * 4 * 256 * KV_BPE_Q8_0   # (K+V)×kv_heads×head_dim = 2176 B/token/FA 层
GDN_LAYER_BYTES = 315_147_136     # GDN 块 14 张量 packed 字节 (gguf_dump n_bytes 实测)
FA_LAYER_BYTES = 305_399_808      # FA 块 11 张量 packed 字节 (gguf_dump n_bytes 实测)
TOKEN_EMBD_BYTES = 1_350_860_800  # token_embd Q8_0 [248320,5120] (仅首 stage)
OUTPUT_NORM_BYTES = 20_480        # output_norm F32 [5120] (仅末 stage; 账本闭环时记 stage0,
                                  # 实际绑定末 stage — weight_loader _collect_names is_last)
MTP_BLOCK_BYTES = 451_319_808     # MTP blk.64 15 张量 Q8_0+F32 (仅末 stage)
LMHEAD_BYTES = 1_350_860_800      # output.weight Q8_0 [248320,5120] (仅末 stage)
HIDDEN_SIZE = 5120                # gguf qwen35.embedding_length
FFN_SIZE = 17408                  # gguf qwen35.feed_forward_length
VOCAB_SIZE = 248320               # token_embd 行数
KV_HEADS = 4                      # gguf qwen35.attention.head_count_kv
HEAD_DIM = 256                    # gguf qwen35.attention.key_length/value_length
SSM_STATE_BYTES = 48 * 128 * 128 * 4   # GDN 循环状态 [n_v=48,dk=128,dv=128] f32 = 3,145,728 B (常数)
DEFAULT_MAX_BATCH = 4             # 引擎 max_num_seqs 默认
DEFAULT_MAX_PREFILL_SEQ = 2048    # 单步 prefill token 包络 (E2E 24-token 生成远小于此)
STAGE0_SAFETY_GIB = 15.1          # 4070TiS 16GB 卡安全线 (exploration R2 ⑥: 可用 ~15.1 GiB)
STAGE1_SAFETY_GIB = 21.1          # 2080Ti 22.5GB 卡安全线 (可用 ~21.1 GiB)


def stage_vram_budget(stage_layers: list, full_attention_interval: int = 4,
                      is_first: bool = False, is_last: bool = False,
                      max_batch: int = DEFAULT_MAX_BATCH,
                      max_prefill_seq: int = DEFAULT_MAX_PREFILL_SEQ) -> dict:
    """单 stage 4 维 VRAM 预算 (bytes): W 权重 + D dequant 临时峰 + A 激活峰
    + K KV cache 配额 + G GDN 循环状态。显式公式, 纯算术, CPU 可重算 (self_check k3)。
      W = n_gdn×315,147,136 + n_fa×305,399,808 + stage 专属张量
          (首: token_embd 1,350,860,800; 末: MTP 451,319,808 + LMHead 1,350,860,800
           + output_norm 20,480)
      D = max GEMM 的 bf16 反量化权重 = hidden×ffn×2B (FFN, 非末 stage 峰值);
          末 stage 另比 LMHead vocab×hidden×2B (decode 每步 logits GEMM)
      A = max_batch×max_prefill_seq×(hidden+ffn)×2B (隐藏态 + FFN 中间激活;
          attention 走 flash SDPA, 不物化 seq² 分数矩阵)
      K = n_fa×max_prefill_seq×2176 B (q8_0 KV; 引擎 PPKVCache 若以 bf16 驻留
          则为 2× 此项 — 仍远小于安全线裕度)
      G = n_gdn×3,145,728 B (f32 循环状态, 常数, 不随 seq 增长)
    """
    n_fa = sum(1 for il in stage_layers if is_full_attention(il, full_attention_interval))
    n_gdn = len(stage_layers) - n_fa
    W = n_gdn * GDN_LAYER_BYTES + n_fa * FA_LAYER_BYTES
    if is_first:
        W += TOKEN_EMBD_BYTES
    if is_last:
        W += MTP_BLOCK_BYTES + LMHEAD_BYTES + OUTPUT_NORM_BYTES
    D = HIDDEN_SIZE * FFN_SIZE * 2
    if is_last:
        D = max(D, VOCAB_SIZE * HIDDEN_SIZE * 2)
    A = max_batch * max_prefill_seq * (HIDDEN_SIZE + FFN_SIZE) * 2
    K = n_fa * max_prefill_seq * int(KV_BYTES_PER_TOKEN_FA)
    G = n_gdn * SSM_STATE_BYTES
    return {"W": W, "D": D, "A": A, "K": K, "G": G,
            "total": W + D + A + K + G, "n_fa": n_fa, "n_gdn": n_gdn}


def check_partition_budget(partition: list, capacities_gib: list = None,
                           full_attention_interval: int = 4,
                           max_batch: int = DEFAULT_MAX_BATCH,
                           max_prefill_seq: int = DEFAULT_MAX_PREFILL_SEQ) -> list:
    """逐 stage 校验 4 维预算 ≤ 对应卡安全线。capacities_gib 缺省 = evo-001 异构双卡
    (15.1, 21.1) GiB。任一 stage 越线 → PPLayerPartitionError (PP-PART-006,
    消息含越线 stage 与各维 bytes, 供调用方重新切层)。返回各 stage 预算 dict 列表。"""
    caps = list(capacities_gib) if capacities_gib else [STAGE0_SAFETY_GIB, STAGE1_SAFETY_GIB]
    if len(caps) != len(partition):
        raise PPLayerPartitionError(
            f"PP-PART-006: capacities 给 {len(caps)} 条但 partition 有 {len(partition)} 个 stage")
    GiB = 1024 ** 3
    report = []
    for i, seg in enumerate(partition):
        b = stage_vram_budget(seg, full_attention_interval,
                              is_first=(i == 0), is_last=(i == len(partition) - 1),
                              max_batch=max_batch, max_prefill_seq=max_prefill_seq)
        limit = caps[i] * GiB
        b["limit_gib"] = caps[i]
        b["margin_gib"] = (limit - b["total"]) / GiB
        b["fit"] = b["total"] <= limit
        report.append(b)
        if b["total"] > limit:
            raise PPLayerPartitionError(
                f"PP-PART-006: stage{i} 预算 {b['total'] / GiB:.2f} GiB "
                f"(W={b['W'] / GiB:.2f} D={b['D'] / GiB:.2f} A={b['A'] / GiB:.2f} "
                f"K={b['K'] / GiB:.3f} G={b['G'] / GiB:.3f}) > 安全线 {caps[i]} GiB "
                f"— 需减该 stage 层数或降包络")
    return report


def compute_pp_layer_partition_budgeted(num_layers: int, pp_size: int,
                                        full_attention_interval: int = 4,
                                        vram_ratio: float = 1.0,
                                        capacities_gib: list = None) -> list:
    """Round-4 fix ②: 先按 vram_ratio 切 (28/36 骨架), 再整 4 维 VRAM 预算校验。
    任一 stage 越安全线 → 按预算重推边界 (仅 pp_size=2 支持: 枚举全部 FA 对齐边界,
    保留满足两卡安全线且最接近 vram_ratio 骨架者); 无可行边界 → PPLayerPartitionError。
    evo-001 模型 28/36 满足两卡安全线 (stage0 ~10.01GiB ≤ 15.1 / stage1 ~14.99GiB ≤ 21.1)
    → 原样返回 28/36 (R3 骨架保留)。
    pp_size=1 → 非 evo-001 异构双卡预算范围 → 直通 (不做容量校验)。
    capacities_gib: 各 stage 安全线 (GiB), 缺省 (15.1, 21.1)。"""
    part = compute_pp_layer_partition(num_layers, pp_size, full_attention_interval, vram_ratio)
    if pp_size == 1:
        return part
    try:
        check_partition_budget(part, capacities_gib, full_attention_interval)
        return part
    except PPLayerPartitionError:
        pass
    if pp_size != 2:
        raise PPLayerPartitionError(
            f"PP-PART-006: 预算越线且 pp_size={pp_size} 不支持边界重推 (仅 pp_size=2)")
    # FA 对齐边界枚举: 边界必在 FA 层之后 → split ≡ 0 (mod interval), split ∈ [interval, num_layers-interval]
    best, best_dev = None, None
    for split in range(full_attention_interval,
                       num_layers - full_attention_interval + 1,
                       full_attention_interval):
        cand = [list(range(0, split)), list(range(split, num_layers))]
        try:
            check_partition_budget(cand, capacities_gib, full_attention_interval)
        except PPLayerPartitionError:
            continue
        dev = abs((num_layers - split) / split - vram_ratio)
        if best is None or dev < best_dev:
            best, best_dev = cand, dev
    if best is None:
        raise PPLayerPartitionError(
            "PP-PART-006: 无 FA 对齐边界同时满足两卡 VRAM 安全线")
    return best
