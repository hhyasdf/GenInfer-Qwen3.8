# self_check.py — L0.5 反作弊预检 (verification 运行, implementer 不跑)。
# 运行时验证代码非 no-op: import 成功 + 关键契约符号存在 + 数值真实计算。
import torch
import traceback

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"[self_check] PASS: {name}")
    except Exception as e:
        FAILURES.append(name)
        print(f"[self_check] FAIL: {name} — {e}")
        traceback.print_exc()


def main():
    # 1) 数值基元 (Phase 1)
    def k1():
        from engine.kernels import RMSNorm, fused_add_rms_norm, silu_and_mul
        x = torch.randn(4, 8)
        y, res = RMSNorm(8)(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        si = torch.randn(4, 16)
        out = torch.empty(4, 8)
        silu_and_mul(out, si)
        assert out.shape == (4, 8) and torch.isfinite(out).all()
    check("kernels.rms_norm+silu_and_mul", k1)

    # 2) PP 通信 (Phase 12)
    def k2():
        from engine.pp import (init_pp_process_group, get_pp_group,
                               IntermediateTensors, make_empty_intermediate_tensors,
                               PPError)
        init_pp_process_group()
        g = get_pp_group()
        assert g.pp_size == 1 and g.rank == 0
        it = make_empty_intermediate_tensors(64, "cpu", torch.bfloat16)
        assert it.tensors["hidden_states"].shape[-1] == 64
        # PP-COMM-005: 缺 hidden_states/residual 必须抛 PPError
        try:
            IntermediateTensors({"foo": torch.zeros(1)})
            raise AssertionError("should have raised PPError")
        except PPError:
            pass
    check("pp.intermediate_tensors+init", k2)

    # 3) PP 层分区 (Phase 13a)
    def k3():
        from engine.pp_layer_partition import (compute_pp_layer_partition,
                                               get_pp_indices, is_fa_layer, PPLayerPartitionError)
        # R3: vram_ratio=1.25 对齐不可变门禁 VRAM_RATIO 常量 (非均分 28/36)。
        # Round-2 的 sum(p)==64 对 sizes [28,36] 为 True 掩盖形态缺陷, 对正确的
        # indices 形态反向锁死 — 弱断言已删。
        p = compute_pp_layer_partition(64, 2, vram_ratio=1.25)
        # 形态钉死: 2 个 stage, 每 stage 是层号列表 list[int] (不是扁平 int, 不是 sizes)
        assert len(p) == 2
        assert all(isinstance(s, list) and len(s) > 0 and all(isinstance(l, int) for l in s) for s in p)
        # 精确层号区间 (对照门禁 STAGE0/STAGE1): stage0=blk0..27, stage1=blk28..63
        assert p[0] == list(range(0, 28))
        assert p[1] == list(range(28, 64))
        # 并集 = range(64) 且无重叠 (每层恰好出现一次)
        all_layers = [l for s in p for l in s]
        assert sorted(all_layers) == list(range(64)) and len(set(all_layers)) == 64
        # 非均匀 28/36
        s0, e0 = get_pp_indices(64, 0, 2, partition_list=p)
        s1, e1 = get_pp_indices(64, 1, 2, partition_list=p)
        assert e0 == s1 and e1 == 64
        assert is_fa_layer(3) and not is_fa_layer(0)
        # PP-PART-005: partition_list 和必须==num_layers
        try:
            get_pp_indices(64, 0, 2, partition_list=[32, 30])
            raise AssertionError("should have raised")
        except PPLayerPartitionError:
            pass
        # R4 fix ②: 完整 4 维 VRAM 预算重算 (代码显式公式, 非注释估算) —
        # 28/36 必须满足两卡安全线 (15.1 / 21.1 GiB); D 维 = 每-GEMM bf16 反量化
        # 权重 (fix ① 低温路径), 不再是 R3 的 f32+int64 ~4.5GB。
        from engine.pp_layer_partition import (check_partition_budget,
                                               compute_pp_layer_partition_budgeted)
        budgets = check_partition_budget(p)
        assert len(budgets) == 2 and all(b["fit"] for b in budgets)
        GiB = 1024 ** 3
        assert budgets[0]["total"] <= 15.1 * GiB, \
            f"stage0 预算 {budgets[0]['total'] / GiB:.2f} GiB > 15.1"
        assert budgets[1]["total"] <= 21.1 * GiB, \
            f"stage1 预算 {budgets[1]['total'] / GiB:.2f} GiB > 21.1"
        # D 维钉死为每-GEMM bf16: stage0 = FFN 5120×17408×2B, stage1 = LMHead 248320×5120×2B
        assert budgets[0]["D"] == 5120 * 17408 * 2
        assert budgets[1]["D"] == 248320 * 5120 * 2
        # 预算化切层与纯切层一致 (28/36 满足安全线 → 骨架保留)
        p_b = compute_pp_layer_partition_budgeted(64, 2, vram_ratio=1.25)
        assert p_b == p
    check("pp_layer_partition", k3)

    # 4) PP KV (Phase 13b)
    def k4():
        from engine.pp_kv import PPKVCache, kv_cache_bytes_per_layer, PPKVLayoutError
        cache = PPKVCache(stage_layers=[28, 29, 30, 31], kv_heads=4, head_dim=256)
        # 本 stage FA 层 = 31 (il%4==3)
        cache.allocate(31, 8)
        k0 = torch.randn(4, 256, dtype=torch.bfloat16)
        v0 = torch.randn(4, 256, dtype=torch.bfloat16)
        cache.write(31, 0, k0, v0)
        rk, rv = cache.read(31, 0)
        assert rk.shape == (4, 256) and rv.shape == (4, 256)
        # GDN 层 (28) 无 KV, 但有 recurrent state
        st = cache.gdn_state(28)
        assert st.shape == (48, 128, 128)
    check("pp_kv", k4)

    # 5) PP 调度 (Phase 14)
    def k5():
        from engine.pp_scheduler import estimate_pipeline_bubble, align_chunked_prefill
        assert abs(estimate_pipeline_bubble(2, 4) - 0.25) < 1e-9
        chunks = align_chunked_prefill(100, 32)
        assert sum(chunks) == 100
    check("pp_scheduler", k5)

    # 6) 模型构造 + 前向 (Phase 5/6, 小型合成 cfg)
    def k6():
        from engine.models import QwenTPConfig, Qwen3ForCausalLMTP
        cfg = QwenTPConfig({
            "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "vocab_size": 256, "max_position_embeddings": 8192, "rms_norm_eps": 1e-6,
            "rope_theta": 1e6,
            "partial_rotary_factor": 0.5, "mrope_section": [3, 3, 2],
            "full_attention_interval": 4, "linear_key_head_dim": 8,
            "linear_num_key_heads": 2, "linear_value_head_dim": 8,
            "linear_num_value_heads": 6, "linear_conv_kernel_dim": 4,
            "attn_output_gate": True, "nextn_predict_layers": 1,
            "tie_word_embeddings": False,
        })
        from engine.pp import init_pp_process_group
        init_pp_process_group()
        m = Qwen3ForCausalLMTP(cfg, pp_rank=0, pp_size=1, stage_layers=[0, 1, 2, 3])
        # Issue 9 (R5 漏检根因): 旧版 `m.to(torch.bfloat16)` 整模型 cast 把 f32 norm
        # 权重一并转 bf16 → 无 dtype 失配 → fused 内核 weight-dtype 守卫从未被验证
        # (合成自检弱于且分叉于生产 dtype 流)。改为模仿生产绑定路径 (weight_loader 逐
        # 张量, 无模型级 cast): 线性权重/激活流 bf16, F32 norm 权重保持 f32 (GGUF F32
        # norm 绑为 f32 Parameter)。修复后 k6 复现 fused 路径 dtype 守卫 — f32 weight
        # + bf16 input → _can_fused 守卫失配 → fallback f32 累加参考路径, 不崩。
        m = m.to(torch.bfloat16)
        from engine.kernels import RMSNorm
        for module in m.modules():
            if isinstance(module, RMSNorm):
                module.weight.data = module.weight.data.to(torch.float32)
        m.eval()
        ids = torch.tensor([[1, 2, 3]])
        out = m(input_ids=ids)
        assert out[0] is not None and out[0].shape[-1] == 256
    check("model.forward(synthetic)", k6)

    # 7) 框架 (Phase 8)
    def k7():
        from engine.framework import (Sequence, SequenceStatus, Scheduler,
                                      Sampler, ModelRunner)
        s = Sequence("r1", [1, 2, 3])
        assert s.status == SequenceStatus.WAITING and s.kv_len == 3
        s.sampling_params = {"max_tokens": 8, "temperature": 0.0}
        sched = Scheduler()
        r = sched.schedule([s], [], 100)
        assert r["is_prefill"] and r["batch"] == [s]
        out = Sampler().sample(torch.randn(1, 256), {"temperature": 0.0})
        assert len(out) == 1
    check("framework.sequence+scheduler+sampler", k7)

    print(f"\n[self_check] {0 if not FAILURES else 'FAILED'} — failures: {FAILURES}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
