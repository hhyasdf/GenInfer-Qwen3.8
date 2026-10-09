"""Step-by-step debug of the L1 GDN oracle.

Runs the engine's sub-modules (bf16, CUDA) and the f64 reference in lockstep,
comparing intermediates after each component. Reports the first component that
diverges beyond bf16 noise (rel > 0.1). A bug shows up as an O(1) relative
error at the first component where it is introduced; bf16 noise is O(1e-2).
"""
import sys

import torch
import torch.nn.functional as F

MODEL_PATH = "/home/hhy/Models/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF"
LAYER_ID = 0
N_TOKENS = 4


def dequant_q6k(w: torch.Tensor, input_size: int) -> torch.Tensor:
    from minisgl.kernel import q6k

    n = w.shape[0]
    ids = torch.arange(n, dtype=torch.int32, device=w.device)
    y = torch.empty(n, input_size, dtype=torch.bfloat16, device=w.device)
    q6k.q6k_gather(ids, w, y)
    return y


def ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight


def ref_recurrence_f64(q, k, v, g, beta, scale):
    n = q.shape[0]
    Hq, Hv, Sv = q.shape[1], v.shape[1], v.shape[2]
    S = torch.zeros(Hv, Sv, Sv, dtype=torch.float64)
    o = torch.zeros(n, Hv, Sv, dtype=torch.float64)
    for t in range(n):
        for h in range(Hv):
            qk_head = h % Hq
            q_h = q[t, qk_head]
            k_h = k[t, qk_head]
            v_h = v[t, h]
            g_val = torch.exp(g[t, h])
            beta_h = beta[t, h]
            kv = S[h].T @ k_h
            delta = (v_h - g_val * kv) * beta_h
            S[h] = g_val * S[h] + torch.outer(k_h, delta)
            o[t, h] = (S[h].T @ q_h) * scale
    return o, S


def report(name: str, eng: torch.Tensor, ref: torch.Tensor) -> float:
    eng = eng.detach().cpu().double()
    ref = ref.detach().cpu().double()
    diff = (eng - ref).abs()
    mag = ref.abs().max()
    rel = (diff.max() / mag).item() if mag > 0 else 0.0
    flag = "   <<< DIVERGENCE" if rel > 0.1 else ""
    print(f"{name:26s} max|diff|={diff.max().item():12.6f}  "
          f"mean|diff|={diff.mean().item():12.6f}  |ref|max={mag.item():10.4f}  "
          f"rel={rel:.6f}{flag}")
    return rel


def main():
    from minisgl.models.gguf_reader import GgufReader
    from minisgl.models.gguf_weight import find_main_gguf, load_gguf_weight
    from minisgl.models.config import ModelConfig
    from minisgl.models.qwen35 import Qwen35DecoderLayer, GDNState
    from minisgl.utils import torch_dtype
    from minisgl.core import Context, set_global_ctx

    device = "cuda:0"
    torch.cuda.set_device(0)

    # ---- config + weights (same as oracle) ----
    main_shard = find_main_gguf(MODEL_PATH)
    with GgufReader(main_shard) as reader:
        meta = reader.metadata
    config = ModelConfig.from_gguf(meta)
    eps = config.rms_norm_eps

    prefix = f"model.layers.{LAYER_ID}."
    layer_sd = {}
    for k, v in load_gguf_weight(MODEL_PATH, device):
        if k.startswith(prefix):
            layer_sd[k[len(prefix):]] = v if v.dtype == torch.uint8 else v.to(torch.bfloat16)

    with torch_dtype(torch.bfloat16):
        layer = Qwen35DecoderLayer(config, LAYER_ID)
    layer.load_state_dict(dict(layer_sd))
    gdn = layer.attn  # GDNLayer

    # ---- dequantize Q6_K weights to f64 (reference) ----
    W = {}
    q6k_map = {
        "attn_qkv": ("attn.attn_qkv.weight", 5120),
        "attn_gate": ("attn.attn_gate.weight", 5120),
        "ssm_beta": ("attn.ssm_beta.weight", 5120),
        "ssm_alpha": ("attn.ssm_alpha.weight", 5120),
        "ssm_out": ("attn.ssm_out.weight", 6144),
        "mlp_gate": ("mlp.gate.weight", 5120),
        "mlp_up": ("mlp.up.weight", 5120),
        "mlp_down": ("mlp.down.weight", 17408),
    }
    for name, (sd_key, in_sz) in q6k_map.items():
        W[name] = dequant_q6k(layer_sd[sd_key], in_sz).double().cpu()
    W["ssm_conv1d"] = layer_sd["attn.ssm_conv1d"].double().cpu()
    W["ssm_dt"] = layer_sd["attn.ssm_dt"].double().cpu()
    W["ssm_a"] = layer_sd["attn.ssm_a"].double().cpu()
    W["ssm_norm.weight"] = layer_sd["attn.ssm_norm.weight"].double().cpu()
    W["input_layernorm.weight"] = layer_sd["input_layernorm.weight"].double().cpu()
    W["post_attention_layernorm.weight"] = layer_sd["post_attention_layernorm.weight"].double().cpu()

    # ---- random input ----
    torch.manual_seed(42)
    x_bf16 = torch.randn(N_TOKENS, config.hidden_size, dtype=torch.bfloat16, device=device)

    # ---- GDN state (fresh, like the oracle) ----
    ctx = Context(page_size=1)
    set_global_ctx(ctx)
    state = GDNState(
        conv_buf=torch.zeros(3, 10240, dtype=torch.bfloat16, device=device),
        S=torch.zeros(48, 128, 128, dtype=torch.float32, device=device),
    )
    ctx.gdn_state = {LAYER_ID: state}

    # ================= ENGINE SIDE (bf16, CUDA) =================
    # 1. input_layernorm
    x_norm_eng, residual_eng = layer.input_layernorm.forward(x_bf16, None)
    x_norm_eng_fresh = x_norm_eng.clone()  # snapshot before any downstream op
    x_bf16_fresh = x_bf16.clone()  # snapshot the input too

    # 2. GDN projections
    qkv_mixed_eng = gdn.attn_qkv(x_norm_eng)
    z_eng = gdn.attn_gate(x_norm_eng)
    beta_eng = torch.sigmoid(gdn.ssm_beta(x_norm_eng))
    alpha_eng = gdn.ssm_alpha(x_norm_eng)
    gate_eng = F.softplus(alpha_eng + gdn.ssm_dt) * gdn.ssm_a

    # 3. conv1d + silu
    conv_in = torch.cat([state.conv_buf, qkv_mixed_eng], dim=0)
    conv_in = conv_in.unsqueeze(0).transpose(1, 2)
    w = gdn.ssm_conv1d.view(gdn.conv_channels, 1, gdn.conv_kernel)
    conv_out_eng = F.conv1d(conv_in, w, groups=gdn.conv_channels)
    state.conv_buf = conv_in[0, :, -(gdn.conv_kernel - 1):].transpose(0, 1).clone()
    conv_out_eng = F.silu(conv_out_eng).squeeze(0).transpose(0, 1)

    # 4. split + L2 norm
    q_eng = conv_out_eng[:, : gdn.num_k * gdn.head_k].view(N_TOKENS, gdn.num_k, gdn.head_k)
    k_eng = conv_out_eng[:, gdn.num_k * gdn.head_k : 2 * gdn.num_k * gdn.head_k].view(N_TOKENS, gdn.num_k, gdn.head_k)
    v_eng = conv_out_eng[:, 2 * gdn.num_k * gdn.head_k :].view(N_TOKENS, gdn.num_v, gdn.head_v)
    q_eng = F.normalize(q_eng, dim=-1, eps=1e-6)
    k_eng = F.normalize(k_eng, dim=-1, eps=1e-6)

    # 5. recurrence (CUDA kernel)
    from minisgl.kernel import gdn as gdn_kernel
    S = state.S
    q_f = q_eng.float().unsqueeze(0).contiguous()
    k_f = k_eng.float().unsqueeze(0).contiguous()
    v_f = v_eng.float().unsqueeze(0).contiguous()
    g_f = gate_eng.float().unsqueeze(0).contiguous()
    b_f = beta_eng.float().unsqueeze(0).contiguous()
    dst_f = torch.empty(1, N_TOKENS, gdn.num_v, gdn.head_v, device=device, dtype=torch.float32)
    gdn_kernel.gdn_delta_net(
        q_f, k_f, v_f, g_f, b_f, dst_f, S,
        Hv=gdn.num_v, Hq=gdn.num_k, n_tokens=N_TOKENS, n_seqs=1, scale=gdn._scale,
    )
    o_eng = dst_f[0].to(x_bf16.dtype)

    # 6. gated output
    o_norm_eng = gdn.ssm_norm(o_eng.view(-1, gdn.head_v)).view(N_TOKENS, gdn.d_inner)
    o_gated_eng = o_norm_eng * F.silu(z_eng)

    # 7. output projection
    final_eng = gdn.ssm_out(o_gated_eng).clone()  # clone: post norm mutates in-place

    # 8. post norm
    x_norm2_eng, residual_out_eng = layer.post_attention_layernorm.forward(final_eng, residual_eng)

    # 9. MLP
    h_eng = layer.mlp.forward(x_norm2_eng)

    # ================= REFERENCE SIDE (f64) =================
    # CRITICAL: use the ORIGINAL input. The engine's post_attention_layernorm
    # does residual(=x_bf16) += final in-place, so x_bf16 is corrupted by the
    # time we get here. x_bf16_fresh is the pre-mutation snapshot.
    x_f64 = x_bf16_fresh.cpu().double()
    Hq, Hv, Sv, head_v, d_inner = 16, 48, 128, 128, 6144
    scale = 1.0 / (Sv ** 0.5)

    # 1. input_layernorm
    x_norm_ref = ref_rmsnorm(x_f64, W["input_layernorm.weight"], eps)
    residual_ref = x_f64

    # 2. GDN projections
    qkv_mixed_ref = x_norm_ref @ W["attn_qkv"].T
    z_ref = x_norm_ref @ W["attn_gate"].T
    beta_ref = torch.sigmoid(x_norm_ref @ W["ssm_beta"].T)
    alpha_ref = x_norm_ref @ W["ssm_alpha"].T
    gate_ref = F.softplus(alpha_ref + W["ssm_dt"]) * W["ssm_a"]

    # 3. conv1d + silu
    conv_w = W["ssm_conv1d"]
    conv_in_ref = torch.cat([torch.zeros(3, 10240, dtype=torch.float64), qkv_mixed_ref], dim=0)
    conv_out_ref = torch.zeros(N_TOKENS, 10240, dtype=torch.float64)
    for t in range(N_TOKENS):
        for j in range(4):
            conv_out_ref[t] += conv_w[:, j] * conv_in_ref[t + j]
    conv_out_ref = F.silu(conv_out_ref)

    # 4. split + L2 norm
    q_ref = conv_out_ref[:, : Hq * Sv].view(N_TOKENS, Hq, Sv)
    k_ref = conv_out_ref[:, Hq * Sv : 2 * Hq * Sv].view(N_TOKENS, Hq, Sv)
    v_ref = conv_out_ref[:, 2 * Hq * Sv :].view(N_TOKENS, Hv, head_v)
    q_ref = F.normalize(q_ref, dim=-1, eps=1e-6)
    k_ref = F.normalize(k_ref, dim=-1, eps=1e-6)

    # 5. recurrence
    o_ref, _ = ref_recurrence_f64(q_ref, k_ref, v_ref, gate_ref, beta_ref, scale)

    # 6. gated output
    o_norm_ref = ref_rmsnorm(o_ref.view(-1, head_v), W["ssm_norm.weight"], eps).view(N_TOKENS, d_inner)
    o_gated_ref = o_norm_ref * F.silu(z_ref)

    # 7. output projection
    final_ref = o_gated_ref @ W["ssm_out"].T

    # 8. post norm
    residual_ref2 = residual_ref + final_ref
    x_norm2_ref = ref_rmsnorm(residual_ref2, W["post_attention_layernorm.weight"], eps)

    # 9. MLP
    h_ref = (F.silu(x_norm2_ref @ W["mlp_gate"].T) * (x_norm2_ref @ W["mlp_up"].T)) @ W["mlp_down"].T

    # ================= WEIGHT DIAGNOSTICS =================
    # Is the engine's input_layernorm weight the same as the reference's?
    w_eng = layer.input_layernorm.weight.detach().cpu().double()
    w_ref = W["input_layernorm.weight"]
    w_diff = (w_eng - w_ref).abs()
    print("=== Weight diagnostics ===")
    print(f"input_layernorm.weight: max|diff|={w_diff.max().item():.6f}  "
          f"mean|diff|={w_diff.mean().item():.6f}  |ref|max={w_ref.abs().max().item():.4f}  "
          f"n_diff>1e-3={int((w_diff > 1e-3).sum())}/{w_ref.numel()}")

    # Test flashinfer rmsnorm in isolation: engine kernel vs f64, same input+weight.
    from flashinfer import rmsnorm as fi_rmsnorm
    fi_out = fi_rmsnorm(x_bf16, layer.input_layernorm.weight, eps)  # bf16, CUDA
    fi_out_f64 = fi_out.detach().cpu().double()
    fi_diff = (fi_out_f64 - x_norm_ref).abs()
    print(f"flashinfer rmsnorm vs f64: max|diff|={fi_diff.max().item():.6f}  "
          f"mean|diff|={fi_diff.mean().item():.6f}  rel={fi_diff.max().item()/w_ref.abs().max().item():.6f}")

    # Also test with a random weight (rules out weight-specific issues).
    torch.manual_seed(7)
    rand_w = torch.randn(config.hidden_size, dtype=torch.bfloat16, device=device)
    fi_rand = fi_rmsnorm(x_bf16, rand_w, eps)
    ref_rand = ref_rmsnorm(x_f64, rand_w.cpu().double(), eps)
    rand_diff = (fi_rand.detach().cpu().double() - ref_rand).abs()
    print(f"flashinfer rmsnorm (rand weight) vs f64: max|diff|={rand_diff.max().item():.6f}  "
          f"mean|diff|={rand_diff.mean().item():.6f}  rel={rand_diff.max().item()/ref_rand.abs().max().item():.6f}")

    # IN-PLACE MUTATION CHECK: is x_norm_eng (or x_bf16) modified after computation?
    xne_now = x_norm_eng.detach().cpu().double()
    xne_fresh = x_norm_eng_fresh.detach().cpu().double()
    print(f"x_norm_eng vs fresh snapshot: max|diff|={(xne_now - xne_fresh).abs().max().item():.6f}  "
          f"(0.0 => not mutated in-place)")
    xne_fi = (xne_now - fi_out_f64).abs()
    print(f"x_norm_eng vs fi_out (fresh rmsnorm): max|diff|={xne_fi.max().item():.6f}  "
          f"mean|diff|={xne_fi.mean().item():.6f}")
    xb_now = x_bf16.detach().cpu().double()
    xb_fresh = x_bf16_fresh.detach().cpu().double()
    print(f"x_bf16 vs fresh snapshot: max|diff|={(xb_now - xb_fresh).abs().max().item():.6f}  "
          f"(0.0 => input not mutated)")

    # ================= COMPARE =================
    print("=== Step-by-step comparison (engine bf16 vs reference f64) ===")
    report("1. x_norm", x_norm_eng, x_norm_ref)
    report("2. qkv_mixed", qkv_mixed_eng, qkv_mixed_ref)
    report("2. z", z_eng, z_ref)
    report("2. beta", beta_eng, beta_ref)
    report("2. alpha", alpha_eng, alpha_ref)
    report("2. gate", gate_eng, gate_ref)
    report("3. conv_out", conv_out_eng, conv_out_ref)
    report("4. q", q_eng, q_ref)
    report("4. k", k_eng, k_ref)
    report("4. v", v_eng, v_ref)
    report("5. o (recurrence)", o_eng, o_ref)
    report("6. o_norm", o_norm_eng, o_norm_ref)
    report("6. o_gated", o_gated_eng, o_gated_ref)
    report("7. final", final_eng, final_ref)
    report("8. residual", residual_out_eng, residual_ref2)
    report("8. x_norm2", x_norm2_eng, x_norm2_ref)
    report("9. h (MLP)", h_eng, h_ref)


if __name__ == "__main__":
    main()
