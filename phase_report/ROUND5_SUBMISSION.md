# ROUND 5 SUBMISSION — evo-001 MTP token_embd 跨 PP 边界接线

- **Role**: implementer
- **Campaign / Round**: evo-001 / Round 5
- **Timestamp**: 2026-10-20T04:10:39+08:00 (`date -Iseconds`)
- **PID**: 8035（报告头采集进程的 os.getpid()；dsh 宿主下主身份 = 编排器持有的 subagent id，本 PID 仅作交叉佐证）
- **Status**: **SUBMITTED**（implementer 不自证清白 — 不跑 scripts/、不宣判 PASS）
- **改动范围**: `engine/pp.py` (+13/−3), `engine/models/qwen.py` (+66/−12), `engine/framework/runner.py` (+31/−4) — 共 3 文件 +110/−19，git numstat 核验
- **静态自检**: `py_compile` 3 文件 OK；`engine/self_check.py` 7/7 PASS（k6 合成前向已穿过新 MTP 接线，[LOGITS-DIAG] 行在 k6 中实际打印）— **非裁定**

---

## 1. Implemented

### ① 核心修复：MTP token_embd 跨 PP 边界传递（Plan A）

**R4 死点**（不重新诊断，直接接线）：`qwen.py:363` MTP 块只在末 rank 建、`qwen.py:355` token_embd 只在首 rank 绑、`qwen.py:391` 末 rank MTP 前向 `self.token_embd(input_ids)` → `TypeError: 'NoneType' object is not callable` → rank1 崩 → 0 tokens。

**参考语义**（llama.cpp `graph_mtp`, qwen35.cpp:486-644，逐字对齐）：MTP 是**双图输入**——`inp->embd`（输入 token 的 embedding, :512）+ `inp->h`（target 最后层 hidden state, :527）；token_emb 由 `ggml_get_rows(model.tok_embd, inp->tokens)` 行 gather（:519-521）。layer-split 下 embedding 权重在首 rank、MTP 块在末 rank → **token_embd 必须跨首↔末边界走 P2P**（`--spec-draft-device` 只解决 draft 权重共置，解决不了 token_emb 跨边界——这是本轮接线的存在理由）。

**Plan A（采用）vs Plan B（弃用）— 预算依据**：
- Plan A：token_embd 作为**瞬态边界激活**走现有 `send_tensor_dict`/`recv_tensor_dict`（pp.py 既有 P2P 协议，metadata 元组走 CPU gloo，载荷走 NCCL 方向对）。首 rank 打包自身 embedding 输出 [B,T,5120] bf16（prefill T=3 → 30 KiB；decode T=1 → 10 KiB），末 rank MTP 前向读 `it.tensors["token_embd"]`。**零新增常驻权重**：rank1 保持 13.13 GiB（R4 实测），stage1 4 维预算 W+D+A+K+G 一行未动；边界只多一个激活 key，激活维度 A 已覆盖此量级。
- Plan B（弃用）：末 rank 复制 token_embd 权重 ≈ +1.26 GiB 常驻（rank1 12.16+1.26=13.42 GiB < 22.5 GiB，stage1 ≤21.1 GiB 预算内）——可行但更重：多一条 GGUF 权重表复用路径（nextn.embed_tokens 不在权重表）、多 1.26 GiB 常驻、且语义上偏离 llama.cpp（参考实现是共享权重行 gather，不是复制权重）。Plan A 更贴合参考语义、预算更保守，故选 A。

**改动点**：
- `engine/models/qwen.py` forward()：
  - 首 rank 在层消费**前**捕获 `token_emb_local = hidden_states`（embedding 输出 alias）。alias 稳定性依据：`rms_norm.py:53-72` 两条分支（residual=None → `out, x.clone()`；residual 非 None → `res = x + residual` 新张量）均分配新张量、不原地写 x——已逐行核验。FA/GDN/MLP 各层输入均为 F.linear 新张量，不回写上游。
  - MTP 分支：`token_emb = token_emb_local if g.is_first_rank else it.tensors.get("token_embd")`；`None` → 抛 `PPError("PP-COMM-007: ...")`（新错误码，协议失步显式失败，不静默）。**删除** `self.token_embd(input_ids) ... else hidden_states` 旧兜底——⛔ 两条都是错误语义（前者 NoneType 崩，后者 enorm/hnorm 两路同向量 → draft 必错），Issue 8 明令禁止。
  - 边界打包：仅当 `self.token_embd is not None and self.cfg.mtp_layers` 时加第 3 键 `token_embd`（无 MTP 的模型不产生额外边界流量）。全部键 `.contiguous()`。
- `engine/pp.py`：
  - `make_empty_intermediate_tensors` 加 `token_embd` 占位（(1, hidden_size) 与另两键同形）——文档注释 :37 早已声明该缝，现在启用。
  - `send_tensor_dict` metadata 元组的 `tuple(v.shape)`/`v.dtype` 提取加 `isinstance(v, torch.Tensor)` 防护（R4 潜伏缺陷：防护只在 device 一行，非 tensor 值会先在 `.shape` 上 AttributeError；接收侧本就有 `if not is_tensor: continue`，本改动只是与接收侧对称）。当前全 tensor 路径不触发，属防御性修复。
- **generate() 多 rank token 交接**（见 §3 披露）：末 rank 采样后经 `dist.send(tok, dst=g._norm_dst(None))`（rank1 → (1+1)%2 = 0）传回首 rank；首 rank `dist.recv(tok, src=g._norm_src(None))`（rank0 → (0−1)%2 = 1）接收后推进下一步输入。单 rank（pp_size==1 / dist 未初始化）直通，无 P2P。此改动同时消除了原 `if generated:` 本地跟随的失效分支（首 rank 本地 generated 恒空 → 永远拿原 prompt 重复 prefill）。

### ② 引擎侧 logits 有限性诊断（任务 ②）

`qwen.py` 新增模块级 `_logits_finiteness_diag(logits)`，在**末 rank lm_head 之后、return 之前**调用：`torch.isfinite` 全量扫描 + 首个非有限 logit 的位置/数值/占比，单行 `[LOGITS-DIAG]` 输出（可 grep）。**纯只读**：不修改 logits、不改变 greedy argmax 采样路径、不影响任何数值——E2E 第一次到 lm_head 后，Verification 可直接从 L3 日志采集有限性证据。裁定权归 Verification。self_check k6 已实测打印 `finite=768/768 (all finite)`。

---

## 2. Contract Files Read（跨模块调用前先读源 — 铁律 5）

| 文件 | 读取目的 / 关键行 |
|---|---|
| `engine/pp.py` :37/:51-57/:93-97/:102-158 | IntermediateTensors 缝声明；`make_empty` 原 2 键；`_norm_dst/_norm_src` 回绕公式；`send_tensor_dict` metadata 元组 5 字段结构与空张量对称跳过；`recv_tensor_dict` 的 `is_tensor` continue 与按元数据 device 分配缓冲区 |
| `engine/kernels/rms_norm.py` :53-72 | 首 rank alias 稳定性证明（两条分支均新张量，无 in-place 写 x）|
| `engine/models/qwen.py`（全文件）| 死点 363/355/391/398-401/420 的精确原文；MTPBlock.forward 签名 (hidden, token_emb, kv_cache, position_ids)；enorm/hnorm 普通 RMSNorm 索引 [0] |
| `engine/self_check.py` | k2 用 `make_empty(64,"cpu",bf16)` 且只断言 hidden_states → 3 键工厂不破坏 k2；k6 合成前向走 pp_size=1 + MTP 路径 |
| `scripts/test_phase12_pp_communication.py` :62-92 | PP-COMM-003 实为 **hidden_states 末维 shape 断言**（非严格 key-set）；PP-COMM-005 为**超集**断言 `{"hidden_states","residual"} <= keys` → 加 token_embd 键均不触发（门禁冲突疑虑解除，见 Known Issues）|
| `scripts/test_phase12_pp_group_init.sh` :96-127 | 两向往返用同一工厂建 3 键张量 → 收发对称、无 key 数失配、无挂起风险 |
| `engine/framework/runner.py`、`llm_engine.py`、`engine/models/config.py`（mtp_layers=[64]）、`engine/weights/quant_linear.py`（QuantEmbedding 行 gather）、`engine/framework/sequence.py`、`engine/framework/scheduler.py` | 生产路径链路；R4 组件（低温反量化 / meta 占位 / 4 维预算）位置确认 |
| 知识库：`notebooks-cn/08_issues/qwen35_hybrid_27b_gguf.md`（Issue 8 根因+双方案预算+禁向）、`evolution/results/evo-001/AGGREGATE_REPORT_R4.md`、`phase_report/ROUND4_SUBMISSION.md`、`evolution/results/evo-001/exploration_report.md`（graph_mtp 章节 + 边界通信语义章节）、`evolution/strategies/evo-005.json` | R4 实证数据（rank1 13.13 GiB / 28:36 分区 / vram_ratio 1.3757）；禁向清单 |

---

## 3. Self-Diff Review（自读 diff 结论）

`git diff` 全文自读（3 文件 +110/−19），逐 hunk 核对：

1. **MTP 分支三元式**：`token_emb_local if g.is_first_rank else it.tensors.get("token_embd")` — Python 条件表达式惰性求值，pp_size==1 时首 rank 分支只取 token_emb_local，`it` 永不求值 → 无 NameError；pp_size>1 末 rank 时 `it` 必已在 forward 开头 recv 赋值。两路径闭合。
2. **边界打包条件** `self.token_embd is not None and self.cfg.mtp_layers`：无 MTP 模型（k6 合成配置除外，k6 pp_size=1 走首 rank 分支不到打包处）行为与 R4 逐字节一致；有 MTP 时多一个 [B,T,5120] bf16 key。
3. **make_empty 3 键工厂**：self_check k2（只断言 hidden_states）与 phase12 两脚本（对称收发）均不受影响 — 已在静态自检中实证 7/7 PASS。
4. **generate()**：`multi = dist.is_initialized() and g.pp_size > 1` 守卫；dst/src 用 pp.py:93-97 的 `_norm_dst/_norm_src` 回绕（rank1→0 / rank0→1 已手推）；NCCL 配对分析——rank0 每步 [发 3 边界张量 → 收 1 token]，rank1 每步 [收 3 边界张量 → 发 1 token]，两对独立、无环形依赖，不挂起。`position_ids` 的 decode 分支补 `device=cur.device`（原 `torch.tensor([pos])` 落 CPU，mrope 里 CPU×CUDA 会 device mismatch——R4 未触达 decode 故未暴露）。
5. **runner.py**：`get_pp_group()` 包 try/except（phase8/9/10 门脚本以合成模型 + 无 dist 构造 ModelRunner → 走 `_pp_group=None` 直通，表面不变：`run()` 签名、`next_tokens` 返回结构、`request_id` 键全部保留）；prefill position_ids 改逐 token `arange(T, device=self.device)` [3,T]——修复旧代码单一 [3,1] 位置在 T>1 时 mrope cos/sin view 必崩的潜伏问题（R4 L3 驱动绕过 runner 故从未暴露；LLMEngine 生产路径第一次走到 T>1 prefill 必炸，此修复是 ② 诊断可达 lm_head 的前提之一）。
6. **② 诊断**：isfinite 扫描 + 首非有限项定位，纯只读、单行 stdout、`flush=True`；argmax 消费原 logits 不受影响。
7. **R4 组件零触碰证明**：git numstat 仅 3 文件 — `quant_linear.py`（低温反量化 no_grad+bf16-per-GEMM）、`pp_layer_partition.py`（W+D+A+K+G 4 维预算）、meta 占位 + 惰性物化、NCCL 2-rank 边界（metadata device 字段 + 双路 device 放置）、BPE tokenizer、28/36 list[list[int]] 分区均未在 diff 中出现。
8. **禁区零 mtime 变化**：以 R4 提交报告（Oct 19 23:50）为时间基准 `find -newer` — `scripts/` **0** 文件、`$MODEL_DIR` **0** 文件、`notebooks-cn/` 1 文件（Issue-8 文档，mtime 02:50:09，早于本回合首改 03:56:55，系 R4 issue-analyzer 回流更新，非本回合）、`gpu/logs/` 8 个自动生成事务日志（00:01–02:01 段系 R4 verification 的 GPU 事务；最新一条 mtime 04:11:10 为**本会话之外**发起的事务 — 本会话全部命令为 py_compile/git/find/stat/self_check，未调用 gpu_job.sh，见 Known Issues #1）。本回合实际写入仅限 engine/ 3 文件（mtime 03:56–04:00）。

---

## 4. Known Issues（主动披露，不隐藏）

1. **gpu/logs 04:11:10 条目来源**：该事务日志的 mtime 晚于本回合末次代码编辑（04:00:23），但本会话未执行任何 gpu_job.sh — 应为编排器侧或其他角色在本回合窗口内的 GPU 事务（自动生成的日志文件，不属代码交付物）。列出以便编排器对账，不影响 engine/ 交付物完整性。
2. **Plan A 的 token_embd 边界流量与 4 维预算**：该瞬态激活的 VRAM 成本（首 rank 打包 + 末 rank recv 缓冲，各 ≤ [B,T,5120] bf16）落在既有 A 维度内，`pp_layer_partition.py` 预算公式**未改一行** — 但 Verification 的 L3 VRAM 监控若按 R4 数值做严格回归比对，应把边界激活增量（decode 步 10 KiB / prefill 步 30 KiB 量级，远小于 MiB 粒度）解释为预期差异。
3. **generate() P2P token 交接未经端到端运行验证**：本回合铁律禁模型上卡，NCCL send/recv 配对只做了协议手推（同 group、同 device、独立方向对、无环依赖）。风险点：若 Verification 驱动直接锁步调 `model.forward`（R4 已验证的驱动模式），此路径完全不经过；若驱动改用 `model.generate()`，首步即触发 token 交接 — 挂起/错序会表现为 decode 步 0 处 NCCL 超时，而非 TypeError。请 Verification 优先用 R4 同款直接 forward 驱动做 24-token 验收，generate() 路径作为第二验证项。
4. **alias 稳定性的未来脆弱点**：首 rank `token_emb_local = hidden_states` 是层消费前的 alias，安全性依赖 RMSNorm 不原地写输入（rms_norm.py:53-72 本轮已核验）。若未来 kernel 路径引入 in-place（如 vllm fused 变体改语义），MTP 的 token_emb 会被层循环污染而静默出错。已在代码注释标注依据行号，提请 spec-reviewer 关注。
5. **PP-COMM-007 为新错误码**：现有码 001–006，本轮新增 007（MTP 缺 token_embd 键的协议失步显式失败）。不冲突，列出备查。
6. **门禁冲突疑虑（已解除，留档）**：初判 `make_empty` 加 token_embd 键可能触发 PP-COMM-003（"中间边界只含 hidden+residual"）。逐行读脚本后确认 003 实为 **hidden_states 末维 shape 断言**（`hs.shape[-1] == HID`，非 key-set 严格相等），005 为超集断言 → 3 键工厂两者均 PASS。静态自检 7/7 亦实证。
7. **runner.py prefill positions 行为变更**：LLMEngine 路径的 prefill 从"必崩（mrope view）"变为"可运行" — 属行为修复而非语义保持；若 Verification 的 L2 回归含 LLMEngine 路径断言旧崩溃行为（极不可能，但列出），请知悉。

---

## 5. Blockers

无。代码就绪，待 spec-reviewer 契约核验 → verification L0-L3 验收。

**验收预期对照**（Verification 的 L3 门禁）：
- MTP 块在 rank1 正常前向：末 rank 从边界读 token_embd，enorm 输入 = 输入 token embedding（graph_mtp :512 语义），不再 `NoneType not callable`。
- prefill+decode 全程无 TypeError：边界 3 键收发对称（3 边界张量 + 可选 1 token 张量/步，配对见 §3.4）。
- 24 tokens：token 交接链路已闭合（generate() 内 P2P + runner.run 内 P2P 双入口）。
- ② 诊断：每次到达 lm_head 的 forward 打印一行 `[LOGITS-DIAG]`，24-token 运行预期 25 行（1 prefill + 24 decode）。
