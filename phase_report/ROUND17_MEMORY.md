# R17 Memory — FullAttention q/gate per-head interleave + mrope NeOX 修复 (evo-001 r7-R17)

| 字段 | 值 |
|------|-----|
| Timestamp | 2026-10-27T05:45:00+08:00 |
| Status | ✅ DELIVERED |
| Track | 完整串行路径 (impl→spec→verify) + 前置链路自证轮 |
| 身份 impl（dsh subagent） | 未捕获 (阻塞前台 spawn 直接返回最终输出, 会话即结束; 独立干净上下文调用) |
| 身份 spec（dsh subagent） | 未捕获 (同上) |
| 身份 verif | PID 19091 (报告头自报 os.getpid()) |
| 身份 main | DSH 主会话 (天然区别于 subagent) |
| 身份交叉验证 | verif PID=19091 ≠ main; impl/spec 为独立干净上下文调用 (结构性隔离成立; DSH 阻塞 spawn 未回传 id, 降级记录已如实标注) |

## Scripts Passed (L1, 非豁免分母 10 项)
- test_phase5_attention_init: 9/9 PASS
- test_phase5_kv_cache_paged: 6/6 PASS
- test_phase6_decode_forward_no_clone: 3/3 PASS
- test_phase6_layer_e2e_random_weights: 3/3 PASS
- test_phase6_mlp_forward: 4/4 PASS
- test_phase6_residual_chain: 3/3 PASS
- test_phase7_hf_key_mapping: 4/4 PASS
- test_phase7_qwen_tp_config: 5/5 PASS
- test_phase9_llm_engine_init: 4/4 PASS
- test_phase9_generate_single_gpu: 自 SKIP rc=0 (46.7GB>12GB 单卡预算, R7–R12 基线)
- L2 前序回归: 20/20 rc=0, regression_count=0
- 豁免 2 项 (台账口径): test_phase5_flash_attn_prefill_decode (ENV-EXEMPT, 无 flash_attn → SDPA 路径); test_phase7_weight_loading.sh (Gap-B-EXEMPT, GGUF-only)

## L0.6 Agent Self-Check
- Check 1 (no-op path): PASS
- Check 2 (mock vs real): PASS
- Check 3 (reference gap): PASS
- Check 4 (side-effect visibility): PASS
- Check 5 (boundary injection): PASS

## L3 验收锚 (本轮核心, 全实测)
- il3 (FA) attn_out relmax: 修复前 7.69–10.6 → 修复后跨步最大 3.2e-5 (≥2.4e4× 改善)
- GDN il0–3 链点无回退: attn_out ≤4.7e-4, 链点 absmax ≤1.35e-4
- 24-step forced-ref argmax: 23/24 精确 + s16 引擎=96220 (真 f32 值) vs REF 97884 (q8_0-KV 污染轨迹, 边界 ±1 满足)
- **E2E free-run 贪心对齐: vs 真 f32 参照 (ref_f32_rerun.json) 24/24 完全对齐, first_divergence=None**; vs q8_0 REF 16/24 (s16 后级联, 服务端 KV 量化属性, 非引擎 bug)

## Files Changed
- engine/models/qwen.py (FullAttention 内 2 个 hunk):
  - R16 `_apply_mrope`: adjacent (0::2/1::2) → NeOX (x_rot[...,:half] / [half:2*half], out 写同切片) — 未提交前置态, 本轮核验后保留
  - R17 `forward` q/gate 切分: block-split (q=qg[...,:6144], gate=qg[...,6144:]) → per-head interleave (qg.view(B,T,24,2,256); q=[...,0,:], gate=[...,1,:]) — 根因修复
- 未触碰: GatedDeltaNet / QwenMLP / QuantEmbedding / RMSNorm 类 / 权重加载 / scripts/ (自读 diff 确认)

## 根因链 (诊断 → 修复)
- R16 诊断 (oracle forced 23/24 自证 H4): GDN il0-2/MLP/norm/embedding 逐位排除 (≤1e-5), 首个分歧 = 块3 FA attn_out (relmax 7.69–10.6, normalized 输入逐位匹配)
- R17 链路自证 (R17_CHAIN_SELFVERIFICATION.md): oracle .clone() 为真 bug 修复 (in-place 视图污染, buggy/fixed 差 2.96); s16=96220 与 f32/f16 真参照一致 (q8_0 参照文件 97884 被污染); 初版比对脚本 il0 GDN 5.577 假分歧被旧捕获决定性复现 = 脚本轴/标签错位; 捕获绑定中间态工作树 (mrope 已修, R17 未加), pyc 新鲜无 stale
- R17 根因: qg fused [B,T,12288] 中 q/gate 切分布局错误 — 引擎 block-split vs llama.cpp ggml_view_3d per-head interleave (head h: q=[512h,512h+256), gate=[512h+256,512h+512))
- 旧捕获 (r13, mrope 修复前) FA 分歧与新捕获几乎相同 → 分歧由 q/gate 布局主导, 非 mrope 配对差异 (如实偏差, 不动摇根因域)

## Spot Check
- 抽查脚本: test_phase6_mlp_forward.py
- 主 Agent 亲自 gpu_job.sh 重跑: "PHASE6_MLP_FORWARD: ALL 4 TESTS PASSED" — 与 verification 报告 4/4 PASS 一致 ✅
- llama-server :8080 事务后恢复健康 (PID 19178)

## Errors Encountered
- 引擎侧: 无 (R17 一次通过全门)
- verification 自建 free-run 驱动自身 bug 2 处 (Path 拼接 / PP 语义 rank0 未参与 generate) — 验证者已修复重跑, 与引擎无关
- R17 implementer 探针: R16 比对脚本键名错位 (hs/res2 语义错位) 与轴错位 — 链路自证轮钉死为脚本假象

## 交付裁定
spec-reviewer ✅ PASS (8 子组件逐式核验, 3 条非阻断观察 G-1 paged 措辞 / G-2 causal mask 全局显式化建议 / G-3 f32 手写 norm) + verification ✅ PASS (L0/L0.5/L0.6/L1/L2/L3 全门) → **R17 交付**。
引擎 free-run 贪心输出与真 f32 参照 24/24 完全对齐 — 首个 E2E 对齐里程碑。
下一轮 R18: 按回路 B 协议 → knowledge-consolidator 固化 qwen35 hybrid 知识 (FA q/gate interleave / mrope NeOX / GDN 状态语义 / 单流残差约定 / q8_0 KV 边界属性) 到 notebooks-cn/, 然后全 Phase 覆盖重验。
