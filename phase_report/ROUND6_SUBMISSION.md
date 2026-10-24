# ROUND 6 SUBMISSION — evo-001 目标① MTP plain RMSNorm 融合内核 dtype 失配修复

- **Role**: implementer
- **Campaign / Round**: evo-001 / Round 6（Issue 9 驳回后修复轮，最高优先）
- **Timestamp**: 2026-10-20T08:08:17+08:00（`date -Iseconds` 实测）
- **PID**: 23064（报告头采集进程的 `os.getpid()` 实测；dsh 宿主下主身份 = 编排器持有的 subagent id，本 PID 仅作交叉佐证）
- **Status**: **SUBMITTED**（implementer 不自证清白 — 不跑 scripts/、不宣判 PASS）
- **改动范围**: `engine/kernels/rms_norm.py` (+16/−4), `engine/self_check.py` (+12/−1) — 共 2 文件 +28/−5，git numstat 核验
- **静态自检**: `py_compile` 2 文件 OK；`engine/self_check.py` 7/7 PASS（k6 已穿过生产 dtype 流）；最小 CUDA 守卫自检 4/4 PASS（真实 vllm fused 内核路径）— **非裁定**

---

## 1. 修复目标与根因（不重新诊断，直接修）

**R5 驳回证据**（AGGREGATE_REPORT_R5.md）：L0.5 与 L3 两条独立路径同点崩溃，0 token。崩溃帧 `qwen.py:468 generate → qwen.py:429 self.mtp(...) → qwen.py:349 self.hnorm(hidden)[0] → rms_norm.py:60 _vops.rms_norm(out, x, self.weight, self.eps) → RuntimeError: expected scalar type BFloat16 but found Float`。

**死亡点（编排器现场核对行号，自读源码确认）**：
1. `engine/weights/weight_loader.py:227-231` `_bind` Parameter 分支 `obj.copy_(w.to(obj.dtype))` — RMSNorm.__init__（rms_norm.py:38-42）创建的 f32 `nn.Parameter` 接收 F32 GGUF norm 权重，`.to(f32)` 是 no-op → F32 norm 权重保持 f32 Parameter 进 bf16 生产流。
2. `engine/kernels/rms_norm.py:48-60` `_can_fused(x)` 只查输入 dtype/设备/连续性，不查 `self.weight.dtype` → plain RMSNorm（`not ones_plus`，MTP 块唯一触发点）+ f32 weight + bf16 input → :60 融合内核硬崩。decoder 层 ones_plus=True 走 dtype-agnostic 参考路径免疫。
3. 连带 `engine/self_check.py:140` k6 合成模型 `m.to(torch.bfloat16)` 把 f32 norm 权重一并 cast → 与生产绑定路径 dtype 流分叉（漏检根因）。

---

## 2. Implemented

### 选择的路径：**路径 B（kernel 守卫）为主 + k6 对齐**（组合，对照参考裁决）

**为什么选 B 不选 A（加载侧物化 bf16）**：
- 验收硬约束是"**norm 数值语义与 llama.cpp 一致（f32 累加）**"。llama.cpp 的 RMSNorm 是**原生 F32 算子**——CPU 参考 `ggml_compute_forward_rms_norm`（`/home/hhy/Projects/llama.cpp/ggml/src/ggml-cpu/ops.cpp:3989-4004`）`switch (src0->type)` **只处理 `GGML_TYPE_F32`** 其余 `GGML_ABORT`；fused MUL 变体（ops.cpp:3971-3979）把 weight 读成 `const float *` 全程 f32 计算；CUDA 后端 `ggml_cuda_op_rms_norm`（`ggml/src/ggml-cuda/norm.cu:478-500`）`GGML_ASSERT(src0->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32)`，fused kernel（norm.cu:502-560）断言 weight mul_tensor 也是 F32。即参考引擎 norm 在 f32 累加、输出回输入 dtype。
- **路径 A**（weight_loader 把 norm 物化 bf16）物化后走 vllm fused 内核，其累加精度依赖内核内部实现（bf16 派发），**无法确定性匹配** llama.cpp 的 F32 累加——满足"不崩"但"f32 累加"语义是隐式的、不可证。
- **路径 B**（守卫失配 fallback 参考路径）失配时落到 rms_norm.py:62-63 的 dtype 无关参考路径 `(x.float() * rms * w.float()).to(x.dtype)` —— **显式 f32 累加**（`x.float()` 升 f32 → `pow(2).mean` f32 → `rsqrt` f32 → 乘 f32 → downcast 回输入 dtype），**确定性**同时满足"不崩 + f32 累加匹配 llama.cpp"。
- norm 权重保持 F32（GGUF 原生 norm dtype，weight_loader 的 `w.to(obj.dtype)` 对 F32 本就是 no-op）→ **weight_loader.py 零改动**。
- 对照 vllm 不变量（`layernorm.py:104-116` `weight_dtype = dtype or torch.get_default_dtype()`，weight 按模型 dtype 创建 → weight.dtype==激活 dtype 结构性保证；kernel 按输入类型派发 `weight.data_ptr<scalar_t>()` 于 `layernorm_kernels.cu:211/:222`，fused 变体 :243 显式 `TORCH_CHECK(weight.scalar_type()==input.scalar_type())`）：本引擎量化驻留路径无模型级 cast，故由**内核门禁守卫**补齐该不变量（`self.weight.dtype == x.dtype`）。

**改动点**：
- `engine/kernels/rms_norm.py` `_can_fused`：staticmethod → 实例方法，守卫追加 `and self.weight.dtype == x.dtype`。失配（f32 weight + bf16 input）→ `forward` 自然 fall through 到 dtype 无关参考路径（f32 累加）。fused 调用点（:60/:68）**零改动**——`not self.ones_plus` 守卫 + 新 weight-dtype 守卫共同把 plain RMSNorm 的 f32 权重挡在 fused 内核外。ones_plus 路径（decoder 层）行为逐字节不变（`not self.ones_plus` 先短路，守卫不触及）。
- `engine/self_check.py` k6：删 `m.to(torch.bfloat16).eval()` 整模型 cast → 改 `m.to(torch.bfloat16)` 后遍历 `m.modules()` 把每个 RMSNorm 的 `weight.data` 重转 f32，`m.eval()`。**模仿生产绑定路径**（weight_loader 逐张量、无模型级 cast）：线性权重/激活流 bf16 + F32 norm 权重保持 f32。自研自检不再弱于且不分叉于生产 dtype 流。

### 保留的 R5 组件（铁律 3 最小侵入，diff 零触碰证明）

git numstat 仅 2 文件——MTP token_embd PP 边界缝合（3-key + PP-COMM-007 守卫，qwen.py:425-429）、低临时反量化（no_grad 逐 GEMM bf16，quant_linear.py）、4 维 VRAM 预算 W+D+A+K+G（pp_layer_partition.py）、构造期 meta 占位 + 惰性物化（quant_linear.py:146/:198）、NCCL 边界 device 放置（pp.py）、BPE tokenizer、分区 28-36、[LOGITS-DIAG] lm_head 后只读有限性扫描（qwen.py:433）**均未在 diff 中出现**。

---

## 3. 改动关键代码段

**rms_norm.py `_can_fused`（守卫落点）**：
```python
def _can_fused(self, x: torch.Tensor) -> bool:
    # ... Issue 9 根因注释 ...
    return (_HAS_VLLM_OPS and x.dtype == torch.bfloat16
            and x.is_contiguous() and x.device.type == "cuda"
            and self.weight.dtype == x.dtype)   # ← 新增守卫
```
失配时 `forward` 的 `(not self.ones_plus) and self._can_fused(x)` 为 False → 落到 `rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps); out = (x.float() * rms * w.float()).to(x.dtype)`（f32 累加）。

**self_check.py k6（生产 dtype 流对齐）**：
```python
m = m.to(torch.bfloat16)
from engine.kernels import RMSNorm
for module in m.modules():
    if isinstance(module, RMSNorm):
        module.weight.data = module.weight.data.to(torch.float32)
m.eval()
```

---

## 4. Contract Files Read（跨模块调用前先读源 — 铁律）

| 文件 | 读取目的 / 关键行 |
|---|---|
| `engine/kernels/rms_norm.py`（全文件 87 行）| `_can_fused` 原 staticmethod 签名与两处调用点（:58 residual=None / :66 residual 非 None）；参考路径 :62-63 f32 累加原文；GatedRMSNorm 继承 super().forward 同守卫 |
| `engine/weights/weight_loader.py`（全文件 232 行）| `_bind` :226-231 Parameter 分支 `obj.copy_(w.to(obj.dtype))` 对 F32 是 no-op 的根因确认；`_map_name` MTP norm → `["mtp","hnorm"/"enorm"/"shared_head_norm","weight"]` 路径 |
| `engine/models/qwen.py` :280-493 | MTPBlock :339-346 全 plain RMSNorm（ones_plus 缺省 False）；:349 hnorm 首崩点；decoder 层 :295-296 ones_plus=True 免疫；:403 token_embd 输出显式 cast bf16（生产流 bf16 来源）；:429 MTP forward |
| `engine/self_check.py`（全文件）| k6 原 :140 `m.to(torch.bfloat16)` 分叉点；k2 make_empty(64,"cpu",bf16) 不受影响 |
| `engine/weights/quant_linear.py`（全文件 239 行）| 合成模型 QuantLinear/QuantEmbedding meta 占位 + 惰性物化（:146/:178-182/:198/:222-227）— k6 无 loader 路径 dtype 沿用模块当前 dtype；确认重转 f32 norm 不触碰量化占位 |
| 参考 llama.cpp `/home/hhy/Projects/llama.cpp/ggml/src/ggml-cpu/ops.cpp:3960-4010`、`ggml/src/ggml-cuda/norm.cu:470-565`、`src/models/qwen35.cpp:525-555` | RMSNorm 原生 F32 语义（CPU switch 只 F32 / CUDA 断言 F32 / fused MUL 读 float*）；qwen35.cpp:540,543 build_norm(hnorm/enorm) F32 权重 |
| 参考 vllm `/home/hhy/Projects/vllm/vllm/model_executor/layers/layernorm.py:95-125`、`csrc/layernorm_kernels.cu:200-250` | weight_dtype 不变量（:104-116）；kernel 按输入类型派发读 weight.data_ptr<scalar_t>()（:211/:222）；fused TORCH_CHECK（:243） |
| 知识库 Issue 9 全文 `notebooks-cn/08_issues/qwen35_hybrid_27b_gguf.md:417-466`、AGGREGATE_REPORT_R5.md、ROUND5_SUBMISSION.md、exploration_report.md:40-60 | 根因 6 条 + 修复方向 4 条 + 应避免方向 3 条；R5 驳回证据；dtype 直方图 F32:360 |

---

## 5. Self-Diff Review（自读 diff 结论）

`git diff` 全文自读（2 文件 +28/−5），逐 hunk 核对：

1. **`_can_fused` staticmethod → 实例方法**：两处调用点（:58/:66）均为 `self._can_fused(...)`，改实例方法后调用不变。`@staticmethod` 装饰器删除。守卫追加 `self.weight.dtype == x.dtype` 短路求值安全（weight 恒存在）。
2. **fused 调用点零改动**：:60/:68 `_vops.rms_norm(out, x, self.weight, self.eps)` 原样——守卫把 f32 weight 挡在 fused 外，fused 内核只在 weight.dtype==bf16 时可达（此时 `weight.data_ptr<scalar_t>()` 类型一致，无 UB）。
3. **ones_plus 路径逐字节不变**：`(not self.ones_plus) and self._can_fused(...)` 中 `not self.ones_plus` 先短路，decoder 层（ones_plus=True）永不触及新守卫 → R5 已验证的主路径数值零变化。
4. **GatedRMSNorm 继承安全**：`super().forward(x, residual)` 走同一守卫 + 参考路径，gate 乘法在参考路径输出之后，dtype 流一致。
5. **k6 重转 f32 不破坏量化占位**：只遍历 `isinstance(module, RMSNorm)` 的模块，QuantLinear/QuantEmbedding 的 meta 占位（`weight.is_meta`）不触碰；`module.weight.data = ...to(f32)` 重赋值 Parameter 底层张量，Parameter 注册不变。
6. **R5 组件零触碰证明**：git numstat 仅 2 文件，quant_linear.py / pp_layer_partition.py / pp.py / qwen.py / runner.py / tokenizer 均不在 diff。
7. **禁区零 mtime 变化**：baseline 07:45（本回合首动作前）`find -newermt` — scripts/ **0**、notebooks-cn/ **0**、$MODEL_DIR **0**；evolution/ 1 文件（`strategies/evo-006.json` 07:46，编排器 spawn 前写的本轮策略文件，非本回合写入）、gpu/ 1 文件（`logs/gpu-job-...-30377.log` 08:07，自动生成的 GPU 事务日志 — 本会话命令仅 py_compile/self_check/nvidia-smi/git/find，未调用 gpu_job.sh，见 Known Issues #1）。本回合实际写入仅 engine/ 2 个 .py（+ 自动 .pyc）。

---

## 6. 自检结果（py_compile / self_check / 最小 CUDA 守卫自检 — 非裁定）

1. **py_compile**：`rms_norm.py` + `self_check.py` → `PY_COMPILE_OK`。
2. **self_check.py 7/7 PASS**（vllm-dev 环境 torch 2.9.1+cu128，全 CPU 张量，不占 llama-server 显存）：
   - k1 kernels.rms_norm+silu_and_mul / k2 pp.intermediate_tensors+init / k3 pp_layer_partition / k4 pp_kv / k5 pp_scheduler / **k6 model.forward(synthetic)** / k7 framework → 全 PASS，`[self_check] 0 — failures: []`。
   - k6 现走生产 dtype 流（f32 norm + bf16 激活），MTP 前向穿过 hnorm/enorm 不崩（dtype 守卫 fallback 参考路径生效）。
   - `[LOGITS-DIAG]` 在 k6 打印 `finite=0/768 value=nan` — **合成模型未初始化权重（torch.empty 垃圾值）的必然结果**，非本修复引入；smoke test 只断言 shape（`out[0].shape[-1]==256`），不断言有限性。
3. **最小 CUDA 守卫自检 4/4 PASS**（真实 vllm fused 内核路径，小张量 2×256 占显存可忽略）：
   - CASE1 f32 weight + bf16 CUDA input → 守卫失配 → 参考路径 → 不崩，输出 bf16 全有限 ✓
   - CASE2 bf16 weight + bf16 CUDA input → 守卫通过 → fused 内核 → 不崩 ✓
   - CASE3 residual 路径 f32 weight + bf16 input → 守卫失配 → 参考路径 → 不崩 ✓
   - CASE4 ones_plus + f32 weight → 恒走参考路径 → 不崩 ✓
   - `HAS_VLLM_OPS: True`（fused 内核真实可达，非 fallback 假通过）。

---

## 7. Known Issues（主动披露，不隐藏）

1. **gpu/logs 08:07 条目来源**：`gpu-job-20261020-060554-30377.log` mtime 晚于 baseline，但本会话未执行任何 gpu_job.sh（命令仅 py_compile/self_check/nvidia-smi/git/find，self_check 全 CPU 张量不占 GPU 事务）— 应为编排器侧或其他角色的 GPU 事务自动生成日志。列出供编排器对账，不影响 engine/ 交付物完整性。
2. **evolution/strategies/evo-006.json 07:46**：编排器 spawn 本 implementer 前写的 R6 策略文件，落在 evolution/ 禁区 mtime 检查窗口内，非本回合写入（本回合只写 GenInfer-Qwen3.8/engine/）。列出备查。
3. **k6 合成路径在 CPU 不触发 fused 内核**：`_can_fused` 含 `x.device.type=="cuda"` 守卫，k6 全 CPU 张量 → 恒走参考路径（非经守卫 fallback，而是 fused 本就不可达）。k6 的价值是 **dtype 流对齐**（f32 norm + bf16 激活，不再整模型 cast 弱于生产），守卫的真实 CUDA 验证由本回合最小 CUDA 自检 4/4 + Verification 的 L0.6 CUDA 检查 + L3 E2E 承担。若 Verification 的 L0.6 合成驱动仍是"整模型 cast bf16"形态，对守卫永久失明（Issue 9 漏检同款），建议其 L0.6 合成模型也模仿生产绑定（f32 norm + bf16 激活，不整模型 cast）。
4. **MTP norm 走参考路径的性能影响**：plain RMSNorm（MTP 5 个 norm）失配时 fallback 到 torch 参考路径（f32 累加）而非 fused 内核。decoder 主路径本就因 ones_plus=True 走参考路径，引擎已承诺该性能水平；MTP 块仅 1 层 5 个 norm，性能代价可忽略。正确性（f32 累加匹配 llama.cpp）优先。若后续要提速，可在 weight_loader 把 MTP norm 物化 bf16（路径 A）+ 确认 fused 内核 f32 累加语义后切回 fused — 但那是性能迭代，非本回合缺陷修复范围。
5. **`self.weight.dtype == x.dtype` 守卫对 GatedRMSNorm 的适用**：GatedRMSNorm 继承同一守卫。若其 weight 在某种路径下是 f32 而输入 bf16，守卫同样 fallback 参考路径，gate 乘法在参考输出之后，dtype 流一致。当前生产 GDN 层 norm 构造未显式传 ones_plus（需 Verification L0.6 确认其 weight dtype 流），列出备查。
6. **参考代码缓存缺失披露**：任务所述 vllm 参考缓存在 `/home/hhy/Projects/InferGen/knowledge/` 与 `ref_projects/vllm/` 均为**空目录**（实地核对，0 文件）。本回合改从运行环境实际安装的 vllm 源码 `/home/hhy/Projects/vllm/`（layernorm.py + csrc/layernorm_kernels.cu）核对参考语义，line 号与 Issue 9 文档引用一致（:104-116 / :211 / :222 / :243）。llama.cpp 参考在 `/home/hhy/Projects/llama.cpp/` 实地核对无误。此为知识库缓存缺口，不影响本修复（参考语义已独立核实）。

---

## 8. 未闭合风险主动披露

- **② logits 有限性裁定仍是 Verification 侧**：本修复只解除 MTP dtype 死点，让 E2E 抵达 lm_head。logits 有限性（output.weight Q8_0 99.16% vocab 行非有限的模型数据属性）的结构性裁定权归 Verification——若 argmax 落非有限 vocab 且 llama-server 同 GGUF 出真词 → 结构性不可对齐 → 按硬指令上报编排器 PAUSE 人类裁决，不循环重试。
- **f32 累加与 llama.cpp 的位级对齐未实证**：参考路径 `x.float()*rms*w.float()` 与 llama.cpp CPU `y[i00] = x[i00] * scale * w[i00]`（全 f32）语义同构，但**位级一致**需 Verification 的 L3 对照 llama-server 逐 token 比较实证。本回合只做语义同构保证，不宣判位级对齐。

---

## 9. Blockers

无。代码就绪，待 Verification L0-L3 验收（L0.5 输出非硬编码两 prompt 差异 token + L3 PP=2 E2E 24 token + ② 独立全扫抵达 lm_head）。

**验收预期对照**（Verification 门禁）：
- MTP hnorm/enorm/input_layernorm/post_attention_layernorm/shared_head_norm 在 bf16 生产流不崩：f32 weight + bf16 input → 守卫失配 → 参考路径（f32 累加），不再 `expected scalar type BFloat16 but found Float`。
- prefill+decode 全程 dtype 干净：decoder 层（ones_plus）与 MTP 块（plain + 守卫）均走 f32 累加参考路径，数值语义与 llama.cpp 原生 F32 RMSNorm 一致。
- 24 token + ② logits 有限性实证（或结构性不可对齐上报人类裁决）。
