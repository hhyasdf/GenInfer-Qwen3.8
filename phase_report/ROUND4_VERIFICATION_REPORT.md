# ROUND4_VERIFICATION_REPORT — evo-001 PP=2 异构双卡 · 双重验证裁定

| 字段 | 值 |
|------|-----|
| Role | verification |
| Timestamp | 2026-10-20T02:15:12+08:00 |
| Round | 4 |
| 身份 | DSH clean-context subagent (VERIFICATION_PID=20069, PARENT_PID=1050, subagent 干净上下文, 未读 implementer 报告/思考) |
| 验收对象 | /home/hhy/Projects/GenInfer-Qwen3.8 (engine/ + llm_engine.py) |
| 挂载 | DSH subagent (clean context)；subagent_fork 已禁用 |
| GPU 事务 | 全部经 gpu/gpu_job.sh（exit 0，无 exit-99）；Agent 模型已恢复 |

## 最终裁定

# ❌ FAIL

L0 ∧ L0.5 ∧ L0.6 ∧ L1(27/27) ∧ L2(0) 全 PASS，唯 L3 的「24 token + greedy_match=true」未达成（0 token，MTP 块 TypeError）。R3 的 prefill 反量化 OOM 死点已被本轮 no_grad + bf16 低温反量化**消除**；R4 暴露 MTP 块跨 PP 边界的新死点。

## L0 防假 PASS —— ✅ PASS (26 模块 + 2 顶层)
全部 engine 模块 `__file__` 解析自 AGENT_INFER_ROOT，零外部泄漏、零影子包。（首轮把 torch/numpy/Path/json 第三方属性误报为泄漏，已用正确判据——逐模块文件路径审计 + 影子包检查——复核修正。）

## L0.5 self_check 反作弊 —— ✅ PASS (7/7, exit 0)
`python engine/self_check.py` 7 项全过，模型事务后已恢复。

## L0.6 Agent 自检 5 条 —— ✅ PASS (5/5)【HARD 指令：NCCL + cuda tensor 真 2-rank 边界注入】
静态 grep + 真 2-rank NCCL/cuda 张量边界注入（l06_nccl_boundary_R4.py）：recv buffer 落 cuda:1（R3 device-placement 修复在位）；数值往返 std=0.9531；空张量对称跳过；all_reduce 抛 PP-COMM-002（双 rank）。gloo 仅用于 CPU pickle 冒烟（上轮教训：gloo 对 device-placement 永久盲区）。

## L1 Phase 门禁脚本 —— ✅ PASS (27/27, FAILS=0)
L1-A 21 个非模型脚本 + L1-B 6 个 phase9/10 模型脚本，各单次 gpu_job 事务。分母 27（9 豁免不入分母：ENV-EXEMPT×4 / STRUCT-EXEMPT×4 / Gap-B-EXEMPT×1，不扩列）。一过率（exempt-caliber）= **100%**。

## L2 跨 Phase 回归 —— ✅ PASS (regression_count=0)
Phase 1–11 全量重跑，零回滚。

## L3 E2E + VRAM 证据 —— ❌ FAIL（0 token, greedy_match 不可判）
- **事务**：单次 gpu_job.sh（suspend llama-server → torchrun --nproc_per_node=2 → restart），非 exit 99，Agent 模型已恢复。
- **权重进显存**：rank0 绑定 **372 tensors / 10.16GB**（stage0=[0..27] 28 层 + token_embd）；rank1 绑定 **494 tensors / 13.13GB**（stage1=[28..63] 36 层 + lm_head + MTP 块）。分区 vram_ratio=1.3757 → 28/36（与 R3 相同，check_partition_budget fit=True）。
- **R3 死点已消除（本轮核心实证）**：no_grad + bf16 低温反量化后，prefill 28/36 层期间 VRAM 稳定 GPU0 **~9972 MiB (9.74 GiB)** / GPU1 **~12093 MiB (11.81 GiB)**，远低于限额（15.99 / 22.5 GiB）。R3 的「prefill 反量化 f32+int64 临时量 ~4.5GB 顶穿 16GB 卡（peak 15975MiB → OOM）」**未复现**。
- **新死点（MTP 块）**：`qwen.py:363` MTP 块只在末 rank 创建，`qwen.py:355` token_embd 只在首 rank。MTP 块（`qwen.py:391`）`self.token_embd(input_ids)` 在 rank1 上为 None → `TypeError: 'NoneType' object is not callable` → rank1 崩 → rank0 边界 send 报 gloo "Connection closed by peer"（受害者）。生产 `generate()` 同样传 input_ids → 同样崩 → **引擎 PP=2 设计缺陷**（MTP 块跨 PP 边界 token_emb 依赖未解决），非验证驱动缺陷。
- **VRAM 全程未破限**：MTP 块触发的跳涨 GPU0 9972→15559 MiB (15.20GiB) / GPU1 12093→20959 MiB (20.47GiB) 仍在限额内（无 OOM）；崩溃是 Python TypeError。
- **token 数 = 0，greedy_match 不可判**（崩在 MTP 块，未及 lm_head、未写 E2E_OUT）。

## ③ NaN/inf 有限性裁定（output.weight）—— ⛔ 阻塞（未及 lm_head）
- **R3 已实证的数据属性（沿用）**：output.weight (248320,5120) Q8_0，blocks=39,731,200；finite=1,210,145,504 (95.18%)，NaN=61,173,375，inf=79,521；非有限 d 块=1,914,153；受影响 vocab 行=246,243/248,320 (99.16%)。`dequant_q8_0` 逐字对照 ggml-quants.c:553-567 忠实 → 模型数据属性，非反量化 bug。
- **本轮裁定**：E2E 崩在 MTP 块（`qwen.py:391`），位于 lm_head（`qwen.py:395`）之前 → **logits 从未被计算，首个非有限 logit 位置/数值本轮无法实证**。③ 被 MTP 缺陷抢先阻塞（替代 R3 的 OOM 阻塞）。
- **结构性对齐假设（未实证）**：torch.argmax 对 NaN 是「NaN 胜出 reduction」，llama.cpp 贪心手比 `NaN > x = False` 跳过 NaN。若 logits 含 NaN，引擎 argmax 大概率落首个 NaN vocab 索引 → 垃圾 token → 结构性不可对齐。须 MTP 修复后实证；若确认结构性不可对齐 → 不循环重试，上报结构性阻塞供编排器 PAUSE 人类裁决。

## 死亡点演进
| 轮次 | 死点 | 证据 | 状态 |
|------|------|------|------|
| R2 | 权重从未进显存（构造期 host OOM SIGKILL） | VRAM peak 883MiB，0 token | R3 已修 |
| R3 | 权重进显存，16GB 卡 prefill 反量化 f32+int64 临时量 OOM | VRAM peak 15.60GiB（顶穿 15.99GiB），0 token | **R4 已修** |
| R4 | 反量化 OOM 已消除，MTP 块跨 PP 边界 token_emb 依赖缺失 | prefill 28/36 层 VRAM 9.74/11.81GiB（未破限）；rank1 MTP TypeError，0 token | **本轮新死点** |

## 裁定依据
L0 ∧ L0.5 ∧ L0.6 ∧ L1(27/27) ∧ L2(0) 全 PASS，但 L3 的「24 token + greedy_match=true」未达成（0 token，MTP 块 TypeError）。任一 FAIL → FAIL。

**最终裁定：FAIL**
- 失败项：L3（MTP 块跨 PP 边界 token_emb 依赖缺失 → rank1 TypeError → 0 token → greedy_match 不可判）
- R3 死点（prefill 反量化 OOM）已修复确认；R4 暴露 MTP 新死点
- 附加关键项：③ output.weight 99.16% vocab 行非有限（模型数据属性）本轮被 MTP 缺陷阻塞未触发，须 MTP 修复后复验
