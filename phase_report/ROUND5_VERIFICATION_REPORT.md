# ROUND5_VERIFICATION_REPORT — evo-001 PP=2 异构双卡 · 双重验证裁定

| 字段 | 值 |
|------|-----|
| Role | verification |
| Timestamp | 2026-10-20T06:15:00+08:00 |
| Round | 5 |
| 身份 | DSH clean-context subagent (VERIFICATION_PID=11006, PARENT_PID=1050, subagent 干净上下文, 未读 implementer 报告/思考) |
| 验收对象 | /home/hhy/Projects/GenInfer-Qwen3.8 (engine/ + llm_engine.py + openai_tp_server.py) |
| 挂载 | DSH subagent (clean context)；subagent_fork 已禁用 |
| GPU 事务 | 全部经 gpu/gpu_job.sh（L0 纯 import 例外）；6 个事务全部非 exit-99；Agent 模型每个事务后已恢复（最终 /health → {"status":"ok"}） |
| 裁定 | **❌ FAIL**（L0.5 与 L3 未过；无部分通过/有条件交付状态） |

## 门禁总表

| 门禁 | 结果 | 要点 |
|------|------|------|
| L0 import 路径泄漏 | ✅ PASS | 20/20 模块解析到 GEN_ROOT，零 shadow 包，exit 0 |
| L0.5 self_check 反作弊 | ❌ FAIL | self_check 7/7 exit0；权重绑定 372/494（10.16/13.13GB）实际在显存；唯一性 218/288 张量 0 违规；**输出非硬编码不可达**（generate() 崩于 MTP hnorm dtype，0 token） |
| L0.6 边界注入 | ✅ PASS 5/5 | 3-key token_embd 数值往返 + 空张量对称跳过 + all_reduce→PP-COMM-002 + make_empty 三键落位 + 合成 forward 正控制 + 缺键→PP-COMM-007 负注入 |
| L1 scripts/ 契约 | ✅ PASS 27/27 | one_pass_rate=1.000；4 个单卡脚本按自身合约自 SKIP（phase9 generate_single_gpu, phase10 greedy_align×2, benchmark BENCH-001） |
| L2 跨 Phase 回归 | ✅ PASS | regression_count=0（phase1–11 非豁免 21/21 独立重跑全 rc=0） |
| L3 PP=2 E2E | ❌ FAIL | 权重在显存；VRAM 337 采样峰值 10.45/13.58 GiB 全程未触限；**0 token，greedy_match 不可判**（崩于 prefill MTP hnorm） |
| ② logits 有限性 | ⛔ 阻塞 | logits 从未被计算（崩在 lm_head 之前），独立全扫无数据；同 R4 阻塞，换防为 dtype 缺陷 |

## R5 修复核验：token_embd 边界缝 —— ✅ 缝级生效

- L0.6 负注入：缺 token_embd 键 → rank1 正确抛 **PP-COMM-007**（守卫就位）。
- L0.6 正控制 + L0.5/L3 真实 forward：qwen.py:429 `self.mtp(hidden_states, token_emb, ...)` 成功传递——**R4 的 `TypeError: 'NoneType' object is not callable` 未复现**，边界第 3 键被消费。
- test_phase12_pp_group_init.sh 独立确认边界契约 `keys=['hidden_states', 'residual', 'token_embd']`。

## R5 新死点（引擎缺陷定位）

**崩溃帧**（两条独立路径同点崩溃）：
- L0.5（生产 generate 路径）：`qwen.py:468 generate → qwen.py:429 self.mtp(...) → qwen.py:349 self.hnorm(hidden)[0] → rms_norm.py:60 _vops.rms_norm → RuntimeError: expected scalar type BFloat16 but found Float`
- L3（verification 驱动直连 forward 循环）：同帧，caret 精确定位 `self.hnorm(hidden)`。

**根因**：`engine/kernels/rms_norm.py:58-60` fused 路径 `_can_fused(x)` 只检查输入 dtype/device/contiguity，不检查 weight dtype。GGUF 独立测量（本轮 read_gguf 直读张量表）：nextn.hnorm.weight / nextn.enorm.weight / nextn.shared_head_norm.weight / blk.64.attn_norm.weight / blk.64.post_attention_norm.weight **全部 F32**——经 weight_loader 绑定为 f32 Parameter；生产 bf16 流（llm_engine 不整体 cast 模型；qwen.py:403 显式 cast token_embd 输出为 bf16）→ vLLM `_C.rms_norm` 要求 weight.dtype == 输入 dtype → 硬崩。decoder 层 norm 因 `ones_plus=True` 走 dtype 无关的 torch 参考路径而免疫；MTP 的 plain RMSNorm 是首个触发点。self_check k6 未捕获因其合成模型 `m.to(bfloat16)` 一并 cast 了 f32 norm 权重（合成路径与生产 dtype 流分叉——参考缺口）。

**修复方向（供 implementer 二选一）**：(a) rms_norm.py fused 路径增加 weight dtype 守卫（失配走参考路径或 cast weight 进 kernel）；(b) weight_loader 绑定 MTP norm 权重时物化到激活 dtype。修复判据 = L0.5 两条 prompt 差异 token + L3 24 token 抵达 lm_head。

## ② 裁定状态（第 4 轮阻塞）

logits 从未被计算 → 我的独立 torch.isfinite 全扫无数据。既非"全有限"也非"argmax 落非有限"，是**不可达 lm_head**。R3 实证的数据属性沿用：output.weight 99.16% vocab 行非有限（246,243/248,320）。结构性对齐假设（torch.argmax NaN-reduction vs llama.cpp NaN-skip）待 MTP dtype 修复后实证；若届时 argmax 落非有限 vocab 而 llama-server 同 GGUF 出真词 → 结构性不可对齐 → 不循环重试，上报结构性阻塞供编排器 PAUSE 人类裁决。**对照事实本轮复确认**：llama-server 同一 GGUF 出词连贯（现取 24 token @ 24.65 tok/s，末 token 2228）。

## VRAM 证据（L3 事务，nvidia-smi 0.5s × 337 样本/卡）

| GPU | peak | 总显存 | 占满比 | 硬限 | 安全线 | 判定 |
|-----|------|--------|--------|------|--------|------|
| GPU0 (4070TiS) | 10703 MiB (10.45 GiB) | 16376 MiB | 65.4% | 15.99 GiB | 15.1 GiB | 未触 |
| GPU1 (2080Ti) | 13901 MiB (13.58 GiB) | 22528 MiB | 61.7% | 22.5 GiB | 21.1 GiB | 未触 |

本轮崩在 MTP 第一个 kernel（hnorm），早于 R4 观察到的 MTP 分配跳涨（15.20/20.47 GiB）→ 峰值更低。无 OOM，崩溃是 dtype RuntimeError。

## 死亡点演进

R2 权重未进显存（host OOM）→ R3 prefill 反量化 OOM（peak 15.60GiB）→ R4 MTP token_embd NoneType → **R5 MTP plain RMSNorm fused 内核 dtype 失配**。四轮单调向 lm_head 推进；每轮修复真实生效，L2 regression_count=0 无回滚。

## 豁免台账（沿用 R4 口径，零扩张）

ENV-EXEMPT×4（phase1 kernel_wrappers .py/.sh, phase5 flash_attn_prefill_decode, phase2 custom_ar_init — flash_attn UNAVAILABLE→SDPA，SOP 7d）；STRUCT-EXEMPT×4（phase11 throughput/profiler ×2 — 23.18GB 单卡结构性不可达）；Gap-B-EXEMPT×1（phase7 weight_loading.sh — safetensors index 不可达，GGUF-only）。非豁免分母 = 36−9 = **27**。

## 裁定与交付

**❌ FAIL**。实施者修复目标：RMSNORM-FUSED-DTYPE（RuntimeError dtype 失配）— `engine/kernels/rms_norm.py:58-60` 或 `engine/weights/weight_loader.py`（MTP norm 绑定）。修复后重验 L0.5 输出非硬编码 + L3 E2E 24 token + ② 独立全扫。

驱动/证据文件（results/evo-001/）：l0_path_check_R5.py, l05_model_checks_R5.py, l06_nccl_boundary_R5.py, l1_driver_A_R5.sh (+retry), l1_driver_B_R5.sh, l2_driver_R5.sh, l3_e2e_driver_R5.py, l3_runner_R5.sh, reference_raw_R5.jsonl, vram_sample_R5.log, benchmarks_R5.jsonl, AGGREGATE_REPORT_R5.md。

verification 驱动自纠记录（不影响引擎裁定）：L0.6 首轮因驱动漏物化合成模型非 meta f32 norm 权重而失败（同型缺陷，驱动侧修正后 5/5 通过）；L1-A 首轮 .sh 脚本被 python 误执行（驱动分发 bug，补跑 2/2 PASS）。
