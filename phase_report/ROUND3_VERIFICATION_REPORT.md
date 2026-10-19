# ROUND3_VERIFICATION_REPORT — evo-001 PP=2 异构双卡 · 双重验证裁定

| 字段 | 值 |
|------|-----|
| Role | verification |
| Timestamp | 2026-10-19T20:20:37+08:00 |
| Round | 3 |
| 身份 | DSH clean-context subagent (VERIFICATION_PID=12947, PARENT_PID=1050, subagent 干净上下文, 未读 implementer 报告/思考) |
| 验收对象 | /home/hhy/Projects/GenInfer-Qwen3.8 (engine/ + llm_engine.py) |
| 挂载 | DSH subagent (clean context)；subagent_fork 已禁用 |
| GPU 事务 | 全部经 gpu/gpu_job.sh（exit 0，无 exit-99）；Agent 模型已恢复 |

## 最终裁定

# ❌ FAIL

L0 ∧ L0.5 ∧ L0.6 ∧ L1(27/27) ∧ L2(0) 全 PASS，唯 L3 的「24 token + greedy_match=true」未达成（0 token，VRAM OOM）。

## L0 防假 PASS —— ✅ PASS 8/8
8 模块从 AGENT_INFER_ROOT 解析，零外部泄漏。（修正角色文件过期接口名：实际导出 `PPKVCache`、self_check 仅 `check`/`main`。）

## L0.5 self_check —— ✅ PASS 7/7 (exit 0)
kernels / pp.intermediate_tensors / pp_layer_partition / pp_kv / pp_scheduler / model.forward synthetic / framework 全过。

## L0.6 Agent 自检 5 条 —— ✅ PASS 5/5【HARD 指令：NCCL + cuda tensor 真 2-rank 边界注入】
L06-INIT / L06-DEVPLACE(recv buffer 落 CUDA:1) / L06-SHAPE-VAL / L06-VAL2 / L06-EMPTY / L06-AR(all_reduce→PP-COMM-002) 全过。
潜在观察（非阻塞）：`engine/pp.py:118` send_tensor_dict 对非 tensor 值无条件 `v.shape` → 注入字符串崩溃；recv 侧已处理 is_tensor=False。生产边界为 tensor-only，非阻塞，记录在案。

## L1 Phase 脚本 —— ✅ PASS 27/27 (FAILS=0)
2 个 critical 硬 FAIL 翻绿（test_phase13_pp_layer_partition.py、test_phase12_pp_group_init.sh）+ 25 其余全过。分母 27（9 豁免不入分母，不扩列）。一过率 100%。

## L2 跨 Phase 回归 —— ✅ PASS count=0
Phase 1–11 全部重跑，零回滚。

## L3 E2E + VRAM 证据 —— ❌ FAIL（0 token, greedy_match=false）

**VRAM 证据（采样 96 次，gpu_job.sh 事务内）**：
- GPU0 (RTX 4070 Ti SUPER 16GB)：**peak = 15975 MiB = 15.60 GiB**（顶穿 15.99GiB 容量 → OOM）
- GPU1 (RTX 2080 Ti 22.5GB)：**peak = 13438 MiB = 13.12 GiB**
- 两卡均 >883MiB（R2 死点「权重从未进显存」已修复）
- 权重绑定：rank0 = 372 tensors / 10.16GB；rank1 = 494 tensors / 13.13GB

**token 输出**：
- tokens_generated = **0**（预期 24）
- 死点：prefill 反量化 `up_proj`（`dequant_q6_K`）时 `torch.OutOfMemoryError`（14.70GiB 已分配 / 0 bytes free）
- greedy_match = **false**（无 E2E_OUT 写出）
- prompt_ids = [108618, 102066, 137351]（与参考一致，tokenize 正常）

**判定**：VRAM 近预算达成（15.60/13.12 GiB ≫ 883MiB），但 24 token + greedy_match=true 未达成 → L3 FAIL。

## NaN/inf 有限性裁定（output.weight）
- **任务前提修正**：披露「10 个非有限 Q8_0 d 块」是前 10 块的部分扫描。全量实测：非有限 d 块 = **1,914,153**；NaN 元素 = 61,173,375；inf 元素 = 79,521；受影响 vocab 行 = **246,243/248,320（99.16%）**。
- **忠实性**：`dequant_q8_0`（d=f16, qs=int8[32], out=d*qs）对照 ggml-quants.c:553-567 忠实 → 非有限值是**模型数据属性**，非引擎反量化 bug。
- **矛盾**：llama-server 用同一 GGUF 产生连贯输出（非有限行几乎可确定为 padding，永不为 argmax）。
- **裁定**：E2E 在 prefill 反量化即 OOM，未及 lm_head，NaN/inf **本轮未被实际触发**。「数据性 NaN + 引擎忠实传播 ≠ 引擎缺陷」成立，但「是否影响 output/logits 有限性」**未能实证**（被 OOM 抢先阻断），须下轮 OOM 修复后复验。

## 根因（供 implementer，非 verification 职责）
分区 `vram_ratio=last_mem/first_mem=1.376` 仅按**权重量**切分（stage0=28 层=10.16GB），未预留 `dequant_q6_K` 的 f32 `out` + int64 `ql/qh` 临时量（stage0 上 ~4.5GB）→ 16GB 卡 10.16GB 权重 + ~4.5GB 临时量 + 激活 ≈ 14.7GB，顶穿 15.99GiB → prefill OOM。修复方向：分区为反量化临时量预留显存，或改用低临时量反量化路径，使 16GB 卡不 OOM。

## 裁定依据
任一 FAIL → FAIL。L3 的 24 token + greedy_match=true 未达成（0 token，VRAM OOM）。

**最终裁定：FAIL**（L1 27/27；L3 VRAM peak 15.60GiB/13.12GiB、token 0、greedy_match false）
