# ROUND6_VERIFICATION_REPORT — evo-001 PP=2 异构双卡 · 双重验证裁定

| 字段 | 值 |
|------|-----|
| Role | verification |
| Timestamp | 2026-10-20T10:43:20+08:00 |
| Round | 6 |
| 身份 | DSH clean-context subagent (VERIFICATION_PID=25601, PARENT_PID=1050, subagent 干净上下文, 未读 implementer 报告/思考) |
| 验收对象 | /home/hhy/Projects/GenInfer-Qwen3.8 (engine/ + llm_engine.py + openai_tp_server.py) |
| 挂载 | DSH subagent (clean context)；subagent_fork 已禁用 |
| GPU 事务 | 全部经 gpu/gpu_job.sh（L0 纯 import 例外）；事务均非 exit-99；Agent 模型每个事务后已恢复（最终 /health → {"status":"ok"}, PID 29214） |
| 裁定 | **❌ FAIL**（L3 硬门禁 greedy_match=false → ② 结构性不可对齐；无部分通过/有条件交付状态） |

## 门禁总表

| 门禁 | 结果 | 要点 |
|------|------|------|
| L0 import 路径泄漏 | ✅ PASS | 20/20 模块 + 8 符号解析到 GEN_ROOT，零 shadow 包，exit 0（驱动侧修正 1 个误列的 legacy 符号后重跑通过） |
| L0.5 self_check 反作弊 | ⚠️ 4/5 | self_check 7/7 exit0；权重绑定 372/494（10.16/13.13GB）实际在显存；唯一性 218/288 张量 0 违规；MTP plain RMSNorm 为生产 f32 态；**输出非硬编码 FAIL**（两 prompt 各 10 token 但均为全 token-0 `'!!!!!!'`——NaN logits 使 argmax 对任意 prompt 都落 vocab 0） |
| L0.6 边界注入 | ✅ PASS 5/5 | 3-key token_embd 数值往返 + 空张量对称跳过 + all_reduce→PP-COMM-002 + make_empty 三键落位 + **fused 守卫缝级正控（R5 死点不崩，f32 weight+bf16 input→参考路径，逐位一致）+ 缺键→PP-COMM-007 负注入** |
| L1 scripts/ 契约 | ✅ PASS 27/27 | one_pass_rate=1.000；4 个单卡/4卡脚本按自身合约自 SKIP（phase9 generate_single_gpu, phase10 greedy_align×2, benchmark BENCH-001） |
| L2 跨 Phase 回归 | ✅ PASS | regression_count=0（phase2–10 非豁免 21/21 独立重跑全 rc=0） |
| L3 PP=2 E2E | ❌ FAIL | 权重在显存 ✅；VRAM 峰值 10.49/18.32 GiB 全程未触限 ✅；**24 token ✅ 但 greedy_match=false**（② 全 NaN logits → argmax 落非有限 vocab 0） |
| ② logits 有限性 | ⛔ 结构性阻断 | logits 首次被计算（R1–R5 从未抵达 lm_head）。独立全扫：24/24 步 finite=0/248320、nan=248320、ratio=1.0，first_nonfinite=step1/vocab0/NaN，argmax 24/24 步落非有限 → structural_nonalign=true |

## R6 修复核验：norm dtype 绑定缝 —— ✅ 缝级生效（R5 死点已除）

- **R5 死点不再复现**：`qwen.py:349 self.hnorm(hidden)` 的 `RuntimeError: expected scalar type BFloat16 but found Float` 未再出现。`rms_norm.py:_can_fused` 新增 `self.weight.dtype == x.dtype` 守卫，失配回退 f32-accumulate 参考路径（匹配 llama.cpp F32-native norm 语义）。
- **L0.6 fused 守卫正控（本轮关键新增）**：合成 Qwen3ForCausalLMTP 走**生产 dtype 流**构建（QuantLinear/QuantEmbedding load_dequantized bf16，其余参数 `.to(device)` 不 cast，f32 参数保持 f32——R5 驱动曾整模型 cast bf16 把守卫掩盖了）。rank0 forward 自动打包 token_embd，rank1 forward 返回有限 logits 且断言 `hnorm._can_fused(bf16 probe)==False`（f32 weight + bf16 input → 参考路径），输出与 f32-accumulate 公式**逐位一致**。这正是 R5 死点在 kernel 级的正控制。
- **self_check k6** 现按生产 dtype 流绑定合成模型（bf16 linear + f32 norm/ssm），fused weight-dtype 守卫被真实触发，7/7 通过。
- **MTP plain-RMSNorm 生产态实测**：hnorm/enorm/shared_head_norm/input_layernorm/post_attention_layernorm 全为 f32（正确生产绑定态）。附加数据属性发现：这些 F32 norm 权重自身携带非有限元素（hnorm 58 / enorm 62 / shared_head_norm 41 / input_layernorm 5 / post_attention_layernorm 61）。

## R6 新死点（② logits 有限性 —— 结构性不可对齐，供人类模型有效性裁定）

**这是 R1–R5 从未抵达的 lm_head logits 的首次计算。** 独立 `torch.isfinite(logits)` 全扫：

- **first non-finite logit：step 1, vocab_idx 0, value=NaN, kind=nan。**
- **24/24 decode 步：finite=0/248320，nan=248320，inf=0，nonfinite_ratio=1.0，argmax=0，argmax_val=NaN，argmax_finite=False。**
- **argmax 落非有限 vocab 行：24/24 步；全有限步：0/24 → structural_nonalign=true。**
- 生成 24×token 0 → `'!!!!!!!!!!!!!!!!!!!!!!!!'`；greedy_match（token-level / text）均 false。

**根因（模型数据属性，非引擎逻辑缺陷，忠实于 GGUF）**：NaN 从两处数据属性源流入 lm_head——
1. **MTP F32 norm 权重**携带 NaN/Inf（每 norm 数十个非有限元素），喂入 MTP block forward，污染 hidden_states；
2. **output.weight (lm_head) Q8_0** 全反量化后 99.16%（246,243/248,320）vocab 行非有限；`dequant_q8_0` 忠实于 ggml-quants.c:553-567，NaN 行是模型自身数据。

**llama.cpp 对照（必写进报告的对照事实）**：llama-server 服务**同一 GGUF**（temperature 0.0，n_predict 24）返回 24 token 真实中文 `"务必使游览者无论站在哪个点上，眼前都是一幅完美的图画"。这一特点主要通过以下四个方面实现：\n`（本轮新抓 reference_raw_R6.jsonl，确定性——与 R5 完全一致）。**同一份权重在 llama.cpp 出真词，在本引擎出全 token-0。**

**结构性不可对齐的机理（llama.cpp 分歧点，供人类裁定）**：
- **采样路径 / NaN 处理**：引擎用 `torch.argmax(logits)`；对 100% NaN 行，NaN 在归约中获胜，`argmax` 返回首个下标（vocab 0）→ 每步发 token 0。llama.cpp 的手写比较 greedy 循环把 `NaN > x` 判为 False（**跳过 NaN**，落在有限值）→ 出真词。这是结构性分歧：logits 行全 NaN 时，argmax 在结构上不可对齐。
- **量化精度**：引擎 Q8_0/Q6_K 逐 GEMM 反量化到 bf16（忠实），NaN 行来自模型数据，非反量化 kernel 引入。
- 按 ② 硬指令：NaN 使 greedy 结构性不可对齐 → **如实上报结构性阻断，不循环重试**（已执行）。

## 驱动侧 bug（与引擎 bug 分判）

1. **L0 驱动**误列 `engine.kernels.rms_norm.rms_norm` 函数符号（legacy TP 布局，本 PP 引擎 kernels/rms_norm.py 只有 RMSNorm/GatedRMSNorm 类）。修正（移除 + 加 GatedRMSNorm）后重跑通过。驱动 bug，非引擎。
2. **L0.5/L3 驱动**用 `torch.isfinite(w).not_()`（AttributeError；应为 `~torch.isfinite(w)`）。两处修正后重跑。驱动 bug。
3. **L3 首轮 host-RAM OOM**（rank1 SIGKILL/-9）：29GB 主机 + 22GB GGUF 页缓存 + 2×~11GB raw 字典，内核在权重绑定期 OOM-kill（anon-rss 13.7GB，total-vm 192GB 为 CUDA VA 保留）。**边缘 host-RAM 环境**（R5 同绑定侥幸通过），非引擎持久缺陷——去 profiler 重试成功。
4. **L3 VRAM 采样器**把逐样本行写到 stdout 而非日志文件（vram_sample_R6.log 仅表头）。外观性持久化 bug；峰值数据已从 job 输出捕获。

## VRAM 证据（L3 硬门禁：权重进显存 + 峰值低于上限）

| GPU | 型号 | nvidia-smi 峰值 | torch.max_memory_allocated | 上限/安全线 | 判定 |
|-----|------|----------------|----------------------------|------------|------|
| GPU0 | 4070TiS 16GB | 10738 MiB = 10.49 GiB | 10.46 GB(prefill)/10.53 GB(end) | 15.99 / 15.1 GiB | ✅ 低于安全线 |
| GPU1 | 2080Ti 22.5GB | 18754 MiB = 18.32 GiB | 15.94 GB(prefill)/16.03 GB(end) | 22.5 / 21.1 GiB | ✅ 低于安全线 |

权重实际驻留显存（rank0 372 张量 10.16GB / rank1 494 张量 13.13GB），峰值全程未触限。`engine.generate()` 生产路径交叉验证：24 token，与手工环**逐位一致**。CompiledFunction=[] cudaGraphLaunch=[]（无 compile/graph）。

## R5 → R6 变化

- R5 死点（MTP hnorm dtype RuntimeError）→ **由 1_norm_dtype_binding 修复**（fused 守卫回退）。forward 完成，24 token 产出。L0.6 fused 守卫正控在 R5 死点 kernel 级确认修复。
- R6 新发现（②）：logits 100% NaN → argmax 落非有限 vocab 0 → greedy_match=false → 结构性不可对齐。R1–R5 因引擎从未抵达 lm_head 而不可达。
- L1/L2 维持 27/27 与 regression=0（修复未引入回归）。

## 裁定与交付物

**最终裁定：❌ FAIL**——L3 硬门禁 greedy_match=false（② 结构性不可对齐）。**非回归**：R5 崩溃已由 norm dtype 修复消除，L0/L0.6/L1/L2 全绿。剩余阻断是模型数据属性（NaN logits）使 greedy 采样结构性不可对齐，按 ② 硬指令如实上报、不循环重试、已写 llama.cpp 对照供人类裁定。

- 汇总报告：`evolution/results/evo-001/AGGREGATE_REPORT_R6.md`
- benchmark：`evolution/results/evo-001/benchmarks_R6.jsonl`（首行 correctness.greedy_match=false）
- ② 全扫证据：`evolution/results/evo-001/engine_output_pp2_R6.json`
- llama.cpp 对照：`evolution/results/evo-001/reference_raw_R6.jsonl`
- 驱动（R6 后缀，自主逻辑）：l0_path_check_R6.py / l05_model_checks_R6.py / l06_nccl_boundary_R6.py / l1_driver_A_R6.sh / l1_driver_B_R6.sh / l2_driver_R6.sh / l3_e2e_driver_R6.py / l3_runner_R6.sh
