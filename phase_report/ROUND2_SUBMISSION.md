# Round 2 Fix — Implementer 提交报告

| 字段 | 值 |
|------|-----|
| Role | implementer-inference (reasoning-framework implementer) |
| Track | 驳回后修复 (D-2..D-6b + ⑦) — 非重建 |
| Status | **SUBMITTED** (非 PASS) |
| 母 Agent | evo-001 evolution-loop orchestrator (DSH subagent 挂载, 干净上下文) |
| 禁区确认 | scripts/ notebooks-cn/ 00_contracts/ gpu/ MODEL_DIR 全部 mtime 早于本轮, 零改动 |

## 本轮修了什么 (相对 Round-1 FAIL)

Round-1 27 文件 PP=2 引擎 FAIL: L1 one_pass_rate 65.7% (豁免后 85.2%), L3 零 token
(rank1 host 反量化 OOM SIGKILL), greedy_match=false。本轮**最小侵入**修 6 缺陷 + 1 潜伏:

| 缺陷 | 根因 | 本轮修复 |
|------|------|---------|
| D-2 | (前轮已修) PP 通信/中间张量 | 保留 Round-1 的 23-PASS-script 组件 |
| D-3 | 层分区缺 vram_ratio + 返回 list[list[int]] 与 self_check k3 契约 (sum==64) 冲突 + is_fa_layer 公式判反 | llm_engine:47 传 vram_ratio (双卡显存比≈1.4→28/36); compute_pp_layer_partition 改回返回 sizes list[int]; is_fa_layer=il%4==3 (对照 config/qwen decoder), 加 is_full_attention 别名 |
| D-4 | (前轮已修) PP KV cache | 保留 |
| D-5 | (前轮已修) PP scheduler | 保留 |
| D-6a | 反量化错误 → 权重全错 → L3 零 token | quant_linear.py + gguf_loader.py 三方对齐 (torch==numpy==编译 C `dequantize_row_q6_K`, 0.0 diff); weight_loader.py 全映射重写 (FA 独立 q/k/v/output+q/k_norm; GDN qkv/gate/ssm_*; MLP 分离 ffn_gate/up; MTP blk.64 15 真名) + 量化驻留绑定 (raw uint8 驻 VRAM, 每 GEMM 反量化, 无 bf16 物化) |
| D-6b | hash 占位 tokenize → L1/L3 零有效 token | llm_engine `_tokenize`/`_detokenize` 改真 BPE (Qwen35BPE, 已从 GGUF 内嵌 tokenizer.ggml 加载, 5 条含中文/数字/空格闭包 OK) |
| ⑦ | 潜伏 (前轮已修) | 保留 |

## 关键代码段 (hunks)

### 1. weight_loader.py — 全映射 + 量化驻留
- `_collect_names()` 按层型精确枚举: GDN 14 张量 / FA 11 张量 + norm / MLP 分离 / MTP 15 (Q8_0 FA-only) / top (token_embd/output/output_norm)。
- `_map_name` 正则 `blk\.(\d+)\.(.+)` → FA/GDN/MTP/top 属性路径。
- `_bind`: QuantLinear/QuantEmbedding → `load_quantized(raw, shape, ggml_type)` (uint8 驻留, 前向时 `_weight(dtype)` 反量化); nn.Parameter → `dequant(raw,n,ggml_type).view(shape).copy_`。
- **骨架模型映射验证**: 866 个真实 GGUF 张量名, 映射未命中 0, 解析失败 0, 未被收集 0。

### 2. quant_linear.py — dequant 三方一致 + 量化驻留
- `dequant_q8_0`: d@0(f16) + qs@2:34(int8), 34B/块。
- `dequant_q6_K`: ql@0 + qh@128 + sc@192 + **d@208**(f16, 块末), 210B/块, 2×128 组循环。
- `dequant()` 扩展 F32/F16/BF16 直接 cast (小张量)。
- `QuantLinear._weight(dtype)`: raw uint8 → dequant → x.dtype (每 GEMM 一次, 无 bf16 物化)。
- 默认 init: kaiming_uniform_(a=√5) + QuantEmbedding normal_(std=0.02), 供 k6 无 loader 前向。

### 3. qwen.py — GDN/FA/MLP 计算正确性
- GDN: conv1d(depthwise kernel=4)+silu; q/k L2norm(eps 1e-6)+q/sqrt(dk); gate=softplus(alpha+dt)·ssm_a(负→decay<1); beta=sigmoid; per-head dv RMSNorm×ssm_norm×silu(z); ssm_out。输出 cast 回 x.dtype 避免 bf16/f32 GEMM 混型。
- FA: wq(12288)=q(6144)+gate(6144); attn=sigmoid(gate)·attn **before** wo; SDPA。
- MLP: gate_proj/up_proj/down_proj 分离, silu(gate)·up→down。
- MTP blk.64: 完整 FA decoder 层 (15 张量 Q8_0)。

### 4. llm_engine.py — 接线
- :47 `compute_pp_layer_partition(num_layers, pp_size, vram_ratio=self._vram_ratio())`。
- `_vram_ratio()`: 双卡 total_memory 末段/首段 (21.1/15.1≈1.397→28/36), 不可探测→1.0。
- `_tokenize`/`_detokenize` → Qwen35BPE (惰性从主 GGUF 加载)。

### 5. gguf_loader.py — numpy dequant
- Q8_0_BLOCK 36→34 (f16 d); Q6_K d@208。
- **元素偏移 bug 修复** (多块时"量级离谱"真因): `_dequant_q8_0` 改 `out[i*32:(i+1)*32]` (原用字节偏移当元素偏移→所有块覆写 out[0:32]); `_dequant_q6_K` 加 `bo=i*QK_K` 块元素偏移。修复后 torch==numpy==C 0.0 diff。

## 自检结果 (implementer 自跑, 非 PASS 宣判)

| 项 | 结果 |
|----|------|
| py_compile (7 文件) | ✅ 全过 |
| import 全引擎 | ✅ clean |
| self_check.py (L0.5 反作弊) | ✅ **7/7 PASS** (kernels/pp/partition/kv/scheduler/model.forward/framework) |
| CPU/bf16 合成前向冒烟 (k6 式, 无 loader) | ✅ prefill+decode finite |
| tokenizer 真实闭包 (5 条含中文) | ✅ detokenize(tokenize(x))==x |
| weight_loader 骨架映射 | ✅ 866 名 0 未命中 |
| 禁区 mtime | ✅ scripts/notebooks-cn/00_contracts/gpu/MODEL_DIR 全部早于本轮 |

## 已知未闭合风险 (供 Verification / Consolidator 裁决)

1. **output.weight 非有限 d 块 (模型数据属性, 非反量化 bug)**: Q8_0 源 GGUF 含 10 个非有限 d 块 (7 NaN + 3 inf), 位于块索引 1429,13487,15096,20951,27835,28047(NaN), 32797,33901(inf)。编译 C `dequantize_row_q6_K`/`dequantize_row_q8_0` 参考确认反量化忠实——NaN×0=NaN 源于源数据, 非我方 bug。首个坏块 1429 的 qs 全 0。这是模型数据, implementer 侧不可修; 可能影响 output/logits 的有限性, 需 Verification 在 L3 判定。
2. **self_check k3 分区断言只查 length+sum, 不查 28/36** (软点): k3 仅 `len(p)==2 and sum(p)==64`, 不校验具体 28/36 分布。本轮已让 compute_pp_layer_partition 返回 sizes [28,36] 满足契约, 但 k3 本身不强制 28/36——若 Verification 要求精确 28/36, 需额外断言。
3. **Q6_K 重量级 (std~1e6) 是此合成模型真实属性**: 编译 C 参考证实反量化忠实, 大量级非 bug, 不再列为未闭合项 (但 L3 数值有限性仍受风险 1 的 output.weight NaN 影响)。

## 文件清单 (本轮改动, 均在 AGENT_INFER_ROOT=/home/hhy/Projects/GenInfer-Qwen3.8)

- `engine/weights/quant_linear.py` — dequant 三方一致 + 量化驻留 + F32/F16/BF16 + 默认 init
- `engine/weights/gguf_loader.py` — 34/210 块尺寸 + numpy 元素偏移修复
- `engine/weights/weight_loader.py` — 全映射重写 + 量化驻留绑定
- `engine/models/qwen.py` — GDN/FA/MLP 计算正确性 + MTP 完整 FA 层
- `engine/tokenizer.py` — (前轮 D-6b 已完成, 本轮接线)
- `engine/pp_layer_partition.py` — is_fa_layer 公式修正 + 别名 + 返回 sizes
- `llm_engine.py` — :47 vram_ratio + 真 BPE tokenize/detokenize
- `engine/self_check.py` — (未改, L0.5 反作弊, 7/7 PASS)
- `phase_report/ROUND2_SUBMISSION.md` — 本报告
