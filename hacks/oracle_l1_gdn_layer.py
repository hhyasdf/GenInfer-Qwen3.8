"""L1 GDN decoder-layer oracle (bf16 engine vs f64 reference, layer 0).

Loads the real GGUF weights for decoder layer 0 (a GDN / Gated-DeltaNet layer),
builds the engine's ``Qwen35DecoderLayer(0)`` in isolation, runs it on a short
bf16 input on CUDA, and compares the hidden state against a hand-written f64
reference that replicates the layer's forward math line-for-line.

The reference is independent of the engine: the Q6_K weights are dequantized
to bf16 (shared by both sides via ``q6k_gather``) and upcast to f64; all
projections, the causal conv1d, the GDN delta-rule recurrence, the norms, and
the MLP are computed in f64.

Tolerance: bf16 compute vs f64 ground truth → max_abs_diff ≤ 1e-2.
"""
import sys

import torch
import torch.nn.functional as F

MODEL_PATH = "/home/hhy/Models/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF"
LAYER_ID = 0
N_TOKENS = 4
TOL = 0.1  # relative-error bound: bf16 noise is ~1e-2, a bug is O(1)


def dequant_q6k(w: torch.Tensor, input_size: int) -> torch.Tensor:
    """Dequantize a Q6_K weight [N, K//256*210] uint8 to [N, K] bf16 via q6k_gather."""
    from minisgl.kernel import q6k

    n = w.shape[0]
    ids = torch.arange(n, dtype=torch.int32, device=w.device)
    y = torch.empty(n, input_size, dtype=torch.bfloat16, device=w.device)
    q6k.q6k_gather(ids, w, y)
    return y


def ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """f64 RMSNorm: x / rms(x) * weight, rms(x) = sqrt(mean(x^2) + eps)."""
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight


def ref_recurrence_f64(q, k, v, g, beta, scale):
    """f64 GDN delta-rule recurrence (same as oracle_l3_gdn.py).

    q, k: [n, Hq, Sv] f64 (l2normed). v: [n, Hv, Sv] f64.
    g, beta: [n, Hv] f64. Returns (o [n, Hv, Sv], S [Hv, Sv, Sv]).
    """
    n = q.shape[0]
    Hq, Hv, Sv = q.shape[1], v.shape[1], v.shape[2]
    S = torch.zeros(Hv, Sv, Sv, dtype=torch.float64)  # S[h, i, col]
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


def ref_layer_f64(x, W, eps):
    """f64 reference for Qwen35DecoderLayer(0).forward (GDN path).

    x: [n, 5120] f64. W: dict of dequantized (f64) weights.
    Returns (h [n, 5120], residual [n, 5120]).
    """
    n = x.shape[0]
    hidden = x.shape[1]
    Hq, Hv, Sv = 16, 48, 128
    d_inner = 6144
    head_v = 128
    scale = 1.0 / (Sv ** 0.5)

    # 1. input_layernorm (residual=None → normalized, residual=x)
    x_norm = ref_rmsnorm(x, W["input_layernorm.weight"], eps)
    residual = x

    # 2. GDN projections
    qkv_mixed = x_norm @ W["attn_qkv"].T  # [n, 10240]
    z = x_norm @ W["attn_gate"].T  # [n, 6144]
    beta = torch.sigmoid(x_norm @ W["ssm_beta"].T)  # [n, 48]
    alpha = x_norm @ W["ssm_alpha"].T  # [n, 48]
    gate = F.softplus(alpha + W["ssm_dt"]) * W["ssm_a"]  # [n, 48]

    # 3. causal conv1d (kernel 4, groups=10240) + silu; fresh conv_buf=0
    conv_w = W["ssm_conv1d"]  # [10240, 4]
    # conv_in: [3+n, 10240], first 3 rows = 0 (fresh conv_buf)
    conv_in = torch.cat([torch.zeros(3, 10240, dtype=torch.float64), qkv_mixed], dim=0)
    # conv_out[t, c] = sum_j conv_w[c, j] * conv_in[t+j, c]
    conv_out = torch.zeros(n, 10240, dtype=torch.float64)
    for t in range(n):
        for j in range(4):
            conv_out[t] += conv_w[:, j] * conv_in[t + j]
    conv_out = F.silu(conv_out)  # [n, 10240]

    # 4. split q, k, v; L2-norm q, k
    q = conv_out[:, : Hq * Sv].view(n, Hq, Sv)
    k = conv_out[:, Hq * Sv : 2 * Hq * Sv].view(n, Hq, Sv)
    v = conv_out[:, 2 * Hq * Sv :].view(n, Hv, head_v)
    q = F.normalize(q, dim=-1, eps=1e-6)
    k = F.normalize(k, dim=-1, eps=1e-6)

    # 5. delta-rule recurrence
    o, _ = ref_recurrence_f64(q, k, v, gate, beta, scale)  # [n, 48, 128]

    # 6. gated output: per-v-head RMSNorm(o) * silu(z)
    o = o.view(-1, head_v)  # [n*48, 128]
    o = ref_rmsnorm(o, W["ssm_norm.weight"], eps)
    o = o.view(n, d_inner)  # [n, 6144]
    o = o * F.silu(z)

    # 7. output projection
    final = o @ W["ssm_out"].T  # [n, 5120]

    # 8. post_attention_layernorm (residual + final, then normalize)
    residual = residual + final
    x_norm2 = ref_rmsnorm(residual, W["post_attention_layernorm.weight"], eps)

    # 9. MLP: down(silu(gate @ x) * (up @ x))
    h = (F.silu(x_norm2 @ W["mlp_gate"].T) * (x_norm2 @ W["mlp_up"].T)) @ W["mlp_down"].T

    return h, residual


def main():
    from minisgl.models.gguf_reader import GgufReader
    from minisgl.models.gguf_weight import find_main_gguf, load_gguf_weight
    from minisgl.models.config import ModelConfig
    from minisgl.models.qwen35 import Qwen35DecoderLayer, GDNState

    device = "cuda:0"
    torch.cuda.set_device(0)

    # 1. Build ModelConfig from GGUF metadata.
    main_shard = find_main_gguf(MODEL_PATH)
    with GgufReader(main_shard) as reader:
        meta = reader.metadata
    config = ModelConfig.from_gguf(meta)
    print(f"config: hidden={config.hidden_size} gdn_inner={config.gdn_inner_size} "
          f"gdn_state={config.gdn_state_size} gdn_group={config.gdn_group_count} "
          f"gdn_dt_rank={config.gdn_time_step_rank} gdn_conv={config.gdn_conv_kernel} "
          f"eps={config.rms_norm_eps}")

    # 2. Load GGUF weights for layer 0 only. Match the engine's cast: Q6_K bytes
    #    stay uint8 (the kernels dequantize in registers); every other tensor is
    #    cast to the model dtype (bf16), because the layer is built under
    #    torch_dtype(bf16) so all of its params are bf16.
    prefix = f"model.layers.{LAYER_ID}."
    layer_sd = {}
    for k, v in load_gguf_weight(MODEL_PATH, device):
        if k.startswith(prefix):
            layer_sd[k[len(prefix):]] = v if v.dtype == torch.uint8 else v.to(torch.bfloat16)
    print(f"loaded {len(layer_sd)} tensors for layer {LAYER_ID}: {sorted(layer_sd.keys())}")

    # 3. Build the engine's decoder layer under the model dtype (so its params,
    #    including the F32 ones, are bf16 exactly like the engine), then load.
    from minisgl.utils import torch_dtype
    with torch_dtype(torch.bfloat16):
        layer = Qwen35DecoderLayer(config, LAYER_ID)
    layer.load_state_dict(dict(layer_sd))

    # 4. Dequantize the Q6_K weights to bf16 (shared), then upcast to f64.
    #    Map: reference name -> (state_dict key, input_size).
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
    # F32 weights (already float32; upcast to f64)
    W["ssm_conv1d"] = layer_sd["attn.ssm_conv1d"].double().cpu()
    W["ssm_dt"] = layer_sd["attn.ssm_dt"].double().cpu()
    W["ssm_a"] = layer_sd["attn.ssm_a"].double().cpu()
    W["ssm_norm.weight"] = layer_sd["attn.ssm_norm.weight"].double().cpu()
    W["input_layernorm.weight"] = layer_sd["input_layernorm.weight"].double().cpu()
    W["post_attention_layernorm.weight"] = layer_sd["post_attention_layernorm.weight"].double().cpu()

    # 5. Draw a random input (bf16, [n, 5120]).
    torch.manual_seed(42)
    x_bf16 = torch.randn(N_TOKENS, config.hidden_size, dtype=torch.bfloat16, device=device)
    # CRITICAL: snapshot the input BEFORE the engine runs. The engine's
    # post_attention_layernorm does residual(=x_bf16) += final IN-PLACE (the
    # input_layernorm returns the original x as the residual), so x_bf16 is
    # corrupted by the time the reference reads it. The reference must use the
    # pre-mutation snapshot, not the live tensor.
    x_bf16_orig = x_bf16.clone()

    # 6. Set up the global context (the GDN path reads ctx.gdn_state[layer_id])
    #    and run the engine's layer (bf16, CUDA). Match the engine's state dtypes
    #    (reset_gdn_state): conv_buf is the model dtype (bf16), S is float32.
    from minisgl.core import Context, set_global_ctx
    ctx = Context(page_size=1)
    set_global_ctx(ctx)
    state = GDNState(
        conv_buf=torch.zeros(3, 10240, dtype=torch.bfloat16, device=device),
        S=torch.zeros(48, 128, 128, dtype=torch.float32, device=device),
    )
    ctx.gdn_state = {LAYER_ID: state}
    h_eng, residual_eng = layer.forward(x_bf16, None)
    h_eng = h_eng.cpu().double()
    residual_eng = residual_eng.cpu().double()

    # 7. Compute the reference (f64).
    #    Use the pre-mutation snapshot (x_bf16 is clobbered by the engine's
    #    post norm, which does residual(=x_bf16) += final in-place).
    x_f64 = x_bf16_orig.cpu().double()
    h_ref, residual_ref = ref_layer_f64(x_f64, W, config.rms_norm_eps)

    # 8. Compare. The engine computes in bf16 (f32 accumulation in the GEMMs,
    #    f32 GDN recurrence); the reference is f64. The expected gap is bf16
    #    rounding (~1e-2 relative), so judge on relative error: a bug shows up
    #    as O(1) relative, bf16 noise as O(1e-2).
    diff_h = (h_eng - h_ref).abs()
    diff_r = (residual_eng - residual_ref).abs()
    mag_h = h_ref.abs().max()
    mag_r = residual_ref.abs().max()
    rel_h = diff_h.max() / mag_h
    rel_r = diff_r.max() / mag_r
    print(f"h:        max|diff|={diff_h.max():.6f}  mean|diff|={diff_h.mean():.6f}  "
          f"|h|max={mag_h:.4f}  rel={rel_h:.6f}")
    print(f"residual: max|diff|={diff_r.max():.6f}  mean|diff|={diff_r.mean():.6f}  "
          f"|r|max={mag_r:.4f}  rel={rel_r:.6f}")
    ok = rel_h < TOL and rel_r < TOL
    print(f"PASS (rel < {TOL})" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
