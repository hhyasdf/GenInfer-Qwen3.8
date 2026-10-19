# Round 4 提交报告 — MetaInfer evo-001 两项修复

| 字段 | 值 |
|------|-----|
| Role | implementer |
| Timestamp | 2026-10-19T23:48:53+08:00 |
| Round | 4 |
| Status | **SUBMITTED**（非 PASS/DONE — 裁定权在 verification） |

## 修复范围（仅 dequant 路径 + 预算公式，R3 已绿组件零改动）

### ① 低温 dequant（top priority，R3 L3 死因）

`engine/weights/quant_linear.py`（+79/−36）：
- `dequant_q8_0(raw, n, dtype=None)`：d f16→f32（无损）+ qs int8 **零拷贝 uint8 view**，按 65536 块分块计算，输出直接落 `dtype`（缺省 f32 = R3 行为，位级不变）。
- `dequant_q6_K(raw, n, dtype=None)`：同构分块；组内按 128 元素 quarter 切分，ql 低/高 nibble 与 qh 2-bit 组全部走 **uint8 view**，q 中间量 int8（−32..31 精确），scale 子索引 `is_=l//16` 与 C 参考一致；输出 slice-assign 进 `dtype` 直出的 out 张量。语义对齐 llama.cpp `dequantize_row_q6_K`（ggml-quants.c:1939-1968）/寄存器级 `dequantize_q6_K`（ggml-cuda/dequantize.cuh:246-266，`y=d·sc[is]·(q−32)`）。
- `dequant(raw, n, ggml_type, dtype=None)`：F32/F16/BF16 零拷贝 view + 可选 cast；Q8_0/Q6_K 透传 dtype。
- `QuantLinear._weight`：`dequant(self._raw, n, self._ggml_type, dtype)` 单次物化（删除旧路径 `.to(dtype)` 双物化）。每次 forward 只 dequant 本 GEMM 的那一张权重（stage0 峰值 FFN 17408×5120×2B ≈ 170MB bf16），用完即释，**无整层 f32/bf16 驻留变体**（failed_directions 已死方向未复活）。纯 torch 重实现，未调用任何 llama.cpp CUDA kernel。
- `QuantEmbedding.forward` 行 gather 与 `weight_loader._bind` Parameter 路径（F32/BF16 小张量）行为不变。

### ② 完整 4 维 VRAM 预算切层公式（代码显式，非注释估算）

`engine/pp_layer_partition.py`（+143/−0，全部新增于文件尾部，既有导出零改动）：
- `stage_vram_budget(stage_layers, FAI=4, is_first, is_last, max_batch=4, max_prefill_seq=2048)`：显式纯算术公式
  **W** = n_gdn×315,147,136 + n_fa×305,399,808 + 专属张量（首：token_embd 1,350,860,800 + output_norm 20,480；末：MTP 451,319,808 + LMHead 1,350,860,800）
  **D** = 每-GEMM bf16 反量化权重峰值 = max(hidden×ffn×2B, 末 stage 另比 LMHead vocab×hidden×2B)
  **A** = max_batch×max_prefill_seq×(hidden+ffn)×2B（flash SDPA，无 seq² 物化）
  **K** = n_fa×max_prefill_seq×2176 B（q8_0 KV，2176 = (K+V)×4×256×1.0625）
  **G** = n_gdn×48×128×128×4（GDN f32 循环状态，常数）
- `check_partition_budget(partition, capacities_gib=None)`：逐 stage 校验 total ≤ 安全线（缺省 15.1/21.1 GiB），越线抛 `PPLayerPartitionError`（PP-PART-006，消息含各维 bytes）。
- `compute_pp_layer_partition_budgeted(...)`：先 vram_ratio 切（28/36），再整预算校验；越线时 pp_size=2 按 FA 对齐边界枚举重推（保留最接近 vram_ratio 且两卡可行者）。
- 预算常量与探索账本逐字节对账闭环（exploration_report.md L614-637：GDN 块 315,147,136 B / FA 块 305,399,808 B / stage0 10,106,769,792 B = 9.4127 GiB / stage1 13,059,751,552 B = 12.1628 GiB / 两 stage + 头 padding = 真实文件 23,177,516,384 B ✅）。

`llm_engine.py`（+9/−3）：`__init__` 改调 `compute_pp_layer_partition_budgeted`，预算明细存 `self.stage_vram`。
`engine/self_check.py`（+18/−0）：k3 扩展 4 维预算重算断言（28/36 ≤ 安全线、D 维钉死为每-GEMM bf16、预算化切层与纯切层一致），检查数仍为 7。

**28/36 骨架保留（预算验证通过）**：
- stage0：W=9.413 D=0.166 A=0.344 K=0.028 G=0.062 → **total 10.012 GiB ≤ 15.1**（裕量 5.09 GiB），n_fa=7 n_gdn=21
- stage1：W=12.163 D=2.368 A=0.344 K=0.036 G=0.079 → **total 14.990 GiB ≤ 21.1**（裕量 6.11 GiB），n_fa=9 n_gdn=27
- FAI=4 边界（blk27/28，FA@27、FA@31 分属两 stage 末端）与 MTP blk.64 归属末 stage 均保持。

## 改动文件清单（git numstat，共 +249/−39）

| 文件 | +/− |
|------|-----|
| engine/weights/quant_linear.py | +79/−36 |
| engine/pp_layer_partition.py | +143/−0 |
| engine/self_check.py | +18/−0 |
| llm_engine.py | +9/−3 |

## 禁写区 mtime 检查（cutoff 2026-10-19T22:55:20+08:00）

- `InferGen/notebooks-cn/` → **0** 个新 mtime 文件 ✅
- `InferGen/notebooks-cn/00_contracts/` → **0** ✅
- `InferGen/scripts/` → **0** ✅
- MODEL_DIR（Qwen3.8-27B-…-DFlash2）→ **0** ✅
- `GenInfer-Qwen3.8/scripts`、`GenInfer-Qwen3.8/gpu` 不存在（trivially clean）✅

## 轻自检结果（implementer 权限内，非 PASS 裁定）

| 检查 | 结果 |
|------|------|
| py_compile 四改动文件 | ✅ OK |
| 干净 import（engine.llm_engine + 新符号） | ✅ OK |
| `engine/self_check.py` | ✅ **7/7 PASS**（k3 含预算重算断言） |
| CPU 合成 raw 位级一致性（旧实现 vs 新实现） | ✅ **26/26 PASS** — q6_K f32 七个尺度（1 块 → FFN 全尺寸 5120×17408 六 chunk，含 d=0/±0/极大 f16 边界）；q6_K bf16 直接 == 旧 f32→bf16 位级一致且输出 dtype 即 bf16（无 f32 整层）；q8_0 f32/bf16 四尺度；dispatcher F32/BF16 行为不变；QuantLinear Q6_K/Q8_0 端到端 GEMM 位级一致；QuantEmbedding 行 gather 不变 |
| GPU | 未占用（llama-server :8080 双卡 15622/21030 MiB 全程未动，`nvidia-smi` 复核；全部自检 CPU-only） |

**自检捕获并修复的真实缺陷（证明测试非 no-op）**：首版低温 dequant 存在两处缺陷——(a) qh 组切片漏了 Q6_K 块内 @128 基址（误读 ql 前 32 字节为 qh），(b) `d` 张量 shape 为 [nb,1] 未 reshape 导致 [c,c,32] 广播错。两者均由位级一致性测试（n=256 / n=768）当场捕获，修复后全矩阵（含 FFN 全尺寸）位级一致。

## 未闭环风险披露

1. **stage1 的 D 维是 LMHead 2.368 GiB**，非 170MB — “dequant 临时峰 ≤170MB”验收为 stage0/FFN 专属；stage1 的 decode 每步 logits GEMM 反量化 [248320,5120] bf16 ≈ 2.37 GiB，已显式计入预算（total 14.99 GiB 仍 ≤ 21.1，裕量 6.11 GiB）。这是纯 torch 重实现的结构性代价，非缺陷。
2. **K 维按 q8_0 KV 配额（2176 B/token/FA 层）**；若运行时 PPKVCache 以 bf16 驻留，K 翻倍仅 +0.062 GiB/stage，两卡裕度（5.09/6.11 GiB）完全吸收。
3. **A 维包络 max_batch=4 × max_prefill_seq=2048**（引擎默认与单步包络）；实际 E2E 24-token 生成激活峰远小于此，A 维是保守上界。
4. 预算常量为探索账本实测值（与真实文件大小对账闭环），若未来 GGUF 换量化版本（如 FFN 层改 Q8_0）需重跑 `check_partition_budget`——该函数已代码化，可重算。
5. 26/26 位级一致在 **CPU** 上验证；CUDA 路径（cuBLAS bf16 GEMM）与 CPU GEMM 的数值行为一致由同输入同权重保证（位级一致的 bf16 权重进同一 GEMM），但 GPU 实测峰 VRAM 仍待 verification 的 L3 采样确认（本 implementer 未占 GPU）。
