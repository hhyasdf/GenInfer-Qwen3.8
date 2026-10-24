# Round 3 Fix — Implementer 提交报告

| 字段 | 值 |
|------|-----|
| Role | implementer-inference (reasoning-framework implementer) |
| Timestamp | 2026-10-19T17:41:45+08:00 |
| Round | 3 |
| Track | 驳回后小范围修复 (R2 驳回反馈 ①②③) — 非重建 |
| Status | **SUBMITTED** (非 PASS) |
| 母 Agent | evo-001 evolution-loop orchestrator (DSH subagent 挂载, 干净上下文) |
| 验收依据 | AGGREGATE_REPORT_R2.md 终审打回 3 项 + exploration_report.md R3 增量 (方向 ①②③) |

## 本轮修了什么 (相对 Round-2 FAIL, 按驳回严重度 L3 → L1)

### ① 构造期占位去全触碰 (L3 死因: 权重从未进显存, 构造期 host OOM SIGKILL)

**死亡点消除** — `engine/weights/quant_linear.py`:
- `QuantLinear.__init__` (:100-110): 删 `torch.empty(out,in)` f32 占位 + `nn.init.kaiming_uniform_` 全触碰 (单层 GDN 实测 1.46GB) 与 `torch.zeros(out)` bias 触碰 → 占位改 `torch.empty(..., device="meta")` (:106-107), 对照 vllm meta 哑模型 (gguf_loader.py:174) — **零 host / 零 VRAM / 零触碰**。
- `QuantEmbedding.__init__` (:149-155): 删 f32 占位 (token_embd 5.09GB) + `nn.init.normal_(std=0.02)` 全触碰 → 同改 meta 占位。
- **惰性物化** (仅无 loader 前向触发, self_check k6 合成冒烟): `_weight` (:128-139) / `QuantEmbedding.forward` (:176-184) / `load_dequantized` (:119-126, :167-174) 首次触碰时 `torch.empty(shape, dtype, device="cpu")` — 仍不触碰, 无 kaiming/normal_/zeros_/ones_。
- **权重真正进显存的落地点** — `engine/weights/weight_loader.py` `_bind` (:219-226): `load_quantized(info["raw"].to(self.device), ...)`。根因: `_raw` 是普通属性 (非 Parameter/buffer), `model.to(device)` 不搬它; 若绑定时不落设备, forward 时 dequant 落 CPU → 与 CUDA 激活 device 错配 → L3 硬检查 "权重必须真正进显存" 仍不通过 (与 R2 同签名)。这是本轮核心硬检查的闭环点。
- 铁律遵守: quant_linear.py 全文件 grep 无 kaiming/normal_/zeros_/ones_ 任何全触碰 init (仅注释提及)。
- host 构造瞬态预算闭环 (对照 exploration R3 方向 ① 账):
  - 构造期 (两 rank 合计) = **0** (meta 占位不落 host/VRAM; 实测 50× FFN 尺寸 (5120×17408, f32=344MB/个) 连续构造 host 峰值增量 0 KB)
  - 加载期: 每 rank 只物化自己 stage (load_gguf_raw 按 names 过滤, stage0 9.41GiB / stage1 12.16GiB, 单 rank 瞬态), 两 rank 并行峰值 21.58GiB < 29GB 机器 RAM; 单张量最大 1.26GiB (token_embd/LMHead)
  - VRAM 驻留 = 21.58GiB (stage0 9.41@16GB / stage1 12.16@22.5GB, 双双放得下)

### ② PP recv 设备放置 (L1 FAIL-2: NCCL 组收 CPU 张量必崩)

`engine/pp.py`:
- 元数据元组补 device 字段 (:115-121): `(key, shape, dtype, device, is_tensor)` — device 只抓 `v.device.type` 不抓 index (对照 vllm TensorMetadata parallel_state.py:65 + 发送侧 :77-86 语义 "receiving side will set the device index")。
- recv 侧两条 `torch.empty` 一并补 device (:153 空张量 latent 路径 + :155 载荷路径): `torch.empty(shape, dtype, device=device)` — 缓冲区按元数据 device 分配 (对照 vllm :919-923)。"cuda" 不带 index → 落在本 rank 当前设备 (NCCL 通信器绑定设备); "cpu" → gloo 组合法 (CPU 门禁 phase12_pp_communication.py 语义不变)。
- 发送侧 payload 协议 (int64 前缀 + uint8 载荷, 张量走 device 组) 与空张量对称跳过规则 (Round-2 D-2 修复) 保留未动。

### ③ self_check k3 形态断言 (L1 FAIL-1: 分区形态契约违例被弱断言掩盖)

- `engine/pp_layer_partition.py` `compute_pp_layer_partition` (:36-88): 返回形态由 sizes `list[int]` [28,36] 改回 **`list[list[int]]` 层号索引** — pp_size==2 分支 `[list(range(0,first_size)), list(range(first_size,num_layers))]` (经 `_to_indices` 辅助), pp_size>2 分支同步转换; docstring/注释同步修正 (原 :70-71 注释 "返回 sizes" 已删)。对照不可变门禁 test_phase13_pp_layer_partition.py: STAGE0=[0..27] / STAGE1=[28..63] 精确层号断言。
- `engine/self_check.py` k3 (:54-69): 删 `assert len(p) == 2 and sum(p) == 64` 弱断言 (对 sizes 形态 True → 掩盖; 对 indices 形态 `list==64` False → 反向锁死)。替换为 4 行层号索引断言: 形态 (2 stage 每段 list[int]) + `p[0]==list(range(0,28))` + `p[1]==list(range(28,64))` + 并集=range(64) 无重叠 — 与门禁逐字一致。调用点补 `vram_ratio=1.25` (对齐门禁 VRAM_RATIO 常量的非均分 28/36; 缺省 1.0 会均分 32/32, 与 STAGE0/STAGE1 断言相悖)。`:57-58` get_pp_indices 双调用与 PP-PART-005 负向断言保留未动 (其索引列表分支 :107-119 校验连续覆盖, 与门禁一致)。
- `llm_engine.py:49`: 注释 `# sizes [28,36]` → `# list[list[int]]: [[0..27],[28..63]]` (代码未动 — get_stage_layers/assign_mtp_device 的索引列表分支天然接受, 跨模块签名已读源码确认)。

## 最小侵入确认 (保留 Round-1/2 全部已修复组件)

- D-2 PP 边界协议 2-rank P2P live (int64 前缀 + uint8 载荷 + 空张量对称跳过) — 保留, 本轮只在元组加字段 + recv 补 device
- D-6a 量化驻留 (dequant 三方一致 torch==numpy==C) — 保留, 未改 dequant 函数
- D-6b BPE tokenizer 对齐 llama-server 3/3 — 保留
- ⑦ is_tp_enabled PP=2 no-op — 保留
- Round-1 全部 23 个 PASS 组件 — 未触碰
- 未改 RMSNorm 的 torch.zeros/ones 小占位 (hidden_size×4B=20KB/个, 语义必需: 恒等 norm, 加载时被 copy_ 覆盖; 不在死亡链上, 触碰预算 <1MB/stage) — 若 Verification 认为违铁律, 属裁决项

## 自检结果 (implementer 自跑, 非 PASS 宣判)

| 项 | 结果 |
|----|------|
| py_compile (6 文件: quant_linear/weight_loader/pp/pp_layer_partition/self_check/llm_engine) | ✅ 全过 |
| import llm_engine 全引擎 | ✅ clean (vllm-dev 环境, torch 2.9.1+cu128) |
| 分区形态: compute_pp_layer_partition(64,2,vram_ratio=1.25) == [[0..27],[28..63]] | ✅ 与门禁 STAGE0/STAGE1 逐字一致 |
| get_pp_indices/get_stage_layers/assign_mtp_device 索引分支 | ✅ (0,28)/(28,64)/MTP→stage1 |
| 负向: partition_list=[32,30] sum≠64 仍抛 PPLayerPartitionError | ✅ |
| meta 占位零触碰实证: 50× FFN 尺寸 (f32 344MB/个) 连续构造 host 增量 0 KB | ✅ |
| QuantLinear 量化前向 (Q8_0 合成): 输出与手动 dequant 路径逐位相等, finite | ✅ |
| QuantEmbedding 无 loader 前向: meta 惰性物化成功 | ✅ |
| load_dequantized float 路径 (CPU 冒烟) | ✅ |
| self_check.py (L0.5 反作弊) | ✅ **7/7 PASS** (含新 k3 断言 + k6 无 loader 合成前向) |
| 禁区 mtime | ✅ scripts/ notebooks-cn/ 00_contracts/ gpu/ MODEL_DIR 17:00 后零改动 (find -newermt 为空) |

## 已知未闭合风险 (供 Verification 裁决)

1. **加载期 host 瞬态 = 整 stage 字节** (load_gguf_raw 的 `out` dict 累积 stage 全部 raw 至 load() 返回): 两 rank 并行峰值 21.58GiB < 29GB 物理 RAM, 预算内; 未采用 np.memmap 逐张量流式 (exploration 方向 ① 的理想 1.26GiB 页缓存) — 21.58 < 29 已闭环, 最小侵入不改 load_gguf_raw。若 Verification 实测加载期 RAM 贴边, 属下一轮候选优化, 非本轮缺陷。
2. **PP 组设备路由**: recv 缓冲区 device="cuda" 不带 index, 依赖各 rank 的 torch.cuda 当前设备 = 其 NCCL 通信器绑定设备 (vllm 同款语义)。引擎现未显式 torch.cuda.set_device — 若 L3 驱动层未 set_device 而 NCCL 绑定到卡 0, rank1 缓冲区可能落错卡。llm_engine 的 model.to(cuda:rank) 会设当前设备, 正常驱动流无碍; 列为 Verification L0.6/L3 观察点。
3. **k6 合成前向数值**: meta 惰性物化 = 未触碰垃圾值 (与 Round-2 kaiming 值不同源), k6 只断言形状不断言有限性, 门禁无碍; 但 Verification 合成冒烟若断言 finite, 需知 GDN ssm_a/ssm_dt 本就是 torch.empty 垃圾 (R2 既有行为, harness 侧已按 GGUF 语义初始化过)。
4. **output.weight 非有限 d 块** (R2 风险 1 继承): 模型数据属性 (7 NaN + 3 inf d 块), 非反量化 bug, 可能影响 logits 有限性 — 不变, 需 L3 判定。

## 文件清单 (本轮改动, 均在 AGENT_INFER_ROOT=/home/hhy/Projects/GenInfer-Qwen3.8)

- `engine/weights/quant_linear.py` — meta 占位 + 惰性物化 + 去全触碰 init (修复 ①)
- `engine/weights/weight_loader.py` — _bind raw 落设备 (修复 ① 闭环点)
- `engine/pp.py` — 元数据 device 字段 + recv 两条 torch.empty 补 device (修复 ②)
- `engine/pp_layer_partition.py` — compute_pp_layer_partition 返回 list[list[int]] (修复 ③a)
- `engine/self_check.py` — k3 弱断言 → 4 行层号索引断言 + vram_ratio=1.25 (修复 ③b)
- `llm_engine.py` — :49 注释形态同步 (修复 ③c, 零逻辑改动)
- `phase_report/ROUND3_SUBMISSION.md` — 本报告
