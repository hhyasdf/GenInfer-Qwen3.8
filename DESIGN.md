# DESIGN — qwen38-27b engine

Design decisions for the Qwen3.8-27B GDN-hybrid VLM engine. Every decision cites a
field in `SCENARIO.md`. The scaffold (mini-sglang, CUDA-only) is the starting point;
this document records the deltas the profile demands.

## 1. Overview

- **Model**: Qwen3.8-27B Gated-Delta-Net hybrid VLM (GGUF arch `qwen35`). 64 layers =
  48 GDN (linear-attention) + 16 full-attention (at indices 3,7,11,…,63). Hidden 5120,
  heads 24 / kv 4 / head_dim 256. Context 262144 (model max), vocab 248320.
- **Precision**: main Q6_K, draft Q8_0, mmproj Q8_0 (GGUF-quantized, kept quantized —
  no dequantize-to-bf16; the 54 GB bf16 dequant does not fit 37.31 GiB).
- **Hardware**: 2× PCIe GPU (4070 Ti Super sm_89 16 GB + 2080 Ti sm_86 22 GB), no NVLink.
  37.31 GiB usable. Non-uniform layer split (more on the 2080 Ti).
- **Use case**: OpenAI-compatible HTTP API, single user, consumer = **agent** (unlocks
  constrained decoding), latency-first.
- **Target**: decode ≥ 20 tok/s sustained @ concurrency 1; TTFT(60k) < 30 s.
- **Engine max context**: **100k** (the scenario's floor). 256k does not fit once the
  draft KV cache is counted (40.29 GiB > 37.31 GiB).

## 2. Backend

- **CUDA** (the scaffold is CUDA-only; the profile is 2× NVIDIA GPU).
- **Attention backend seam** (`--attn`): `fi` (FlashInfer) for the **16 full-attn layers**
  (the scaffold's serving default). The **48 GDN layers** do not use the seam — they use a
  ported llama.cpp GDN (delta-net) CUDA kernel (see §6, build-vs-buy). The seam choice
  does not touch the GDN path.
- **Heterogeneous sm_86 + sm_89**: the kernels must compile for both. FlashInfer and the
  ported llama.cpp kernels are sm_86-compatible. Verify at B3 (the scaffold's toy model
  runs on a single GPU; the 2-GPU path is new).

## 3. Model families (deltas from the scaffold)

The scaffold supports llama, mistral, qwen2, qwen3, qwen3_moe. This profile needs three
new families (per `references/models/llm-decoder.md`: "A family that is not in the table
is unsupported until DESIGN.md records its deltas and the class is registered"):

### 3.1 `qwen35` (main model) — GDN hybrid
- **Deltas from qwen3**: 48 of the 64 layers are GDN (linear-attention) instead of
  full-attention. The GDN layer has: a causal conv1d (kernel 4), a delta-rule recurrent
  state (state_size 128, group_count 16, time_step_rank 48, inner_size 6144), and a
  gated output. The 16 full-attn layers are standard multi-head attention with QK-norm
  (Qwen3-style) and head_dim 256.
- **RoPE**: multi-section `[11,11,10,0]`, dim_count 64, base 1e7 (applied to the
  full-attn layers only; the GDN layers use the recurrent state, not RoPE).
- **Activation**: SwiGLU (silu × gate), ffn 17408.
- **Tokenizer**: GGUF-embedded gpt2 BPE (no HF tokenizer.json). Port the llama.cpp vocab
  loader (see §6).
- **Class**: `Qwen35ForCausalLM` (new). Registered in the model registry.

### 3.2 `dflash` (draft model) — speculative-decode draft
- **Deltas**: 5-layer decoder, heads 32 / kv 8, head_dim 128, ffn 17408, ctx 262144.
  Conv layers inside the blocks (attn_conv_base/attn_conv_proj, ffn_conv_base/
  ffn_conv_proj). QK-norm on the attention.
- **Role**: generates up to `n_max = 6` draft tokens per step; the main model verifies
  them (see §7, speculative decoding).
- **Class**: `DflashForCausalLM` (new). Registered in the model registry.

### 3.3 `clip` (mmproj) — Qwen3-VL vision tower
- **Deltas**: 27-layer ViT (embd 1152, ff 4304, 16 heads, patch 16, img 768) + a
  `qwen3vl_merger` projector. Produces image embeddings that are injected into the main
  model's token sequence.
- **Role**: vision input (the origin runs `--mmproj`; the profile includes it). The API
  accepts image input (base64 or URL) alongside text.
- **Class**: `Qwen3VLVision` (new). Registered in the model registry.

## 4. Serving mode

- **http** (OpenAI-compatible): `/v1/chat/completions` (streaming + non-streaming),
  `/v1/completions`, `/v1/models`, `/health`, plus `/admin/*` (the scaffold's control
  surface). The scaffold already has this; no change.
- **consumer = agent** → **constrained decoding** is in scope. The B5 smoke includes one
  tool-call round trip. The engine must accept a tool-call request (with a grammar /
  JSON-schema constraint) and generate a valid tool call. This is a sampling feature (see
  §6, build-vs-buy: constrained decoding).
- **Streaming**: SSE over `/v1/chat/completions` (the scaffold's `StreamingResponse`).

## 5. Batching

- **Concurrency 1** → **single-stream** (one sequence at a time). The scaffold's toy
  model runs at concurrency 1; the real engine keeps single-stream (no continuous
  batching — the profile does not demand it). The scheduler worker streams tokens as they
  are produced (SSE).
- **Chunked prefill**: the load has 60k-token inputs. A single 60k prefill would spike
  the activation workspace and the KV cache. Chunked prefill (ubatch 1024, matching the
  origin's `--ubatch-size 1024`) keeps the per-chunk activation bounded and the TTFT
  predictable. In scope (the load demands it).

## 6. Build vs buy (community-implementations.md)

| Component | Decision | Source / rationale |
|---|---|---|
| Tokenizer | **port** | GGUF-embedded gpt2 BPE; no HF tokenizer.json. Port the llama.cpp vocab loader (`src/vocab.cpp`). No community Python package reads GGUF vocab. |
| Attention (full-attn, 16 layers) | **buy** | FlashInfer (`fi` backend, the scaffold's serving default). |
| Attention (GDN, 48 layers) | **port** | Port the llama.cpp GDN (delta-net) CUDA kernel (`src/models/delta-net-base.cpp` + the GDN kernels in `ggml-cuda`). No community Python package for GDN. |
| GEMM (Q6_K weights) | **port** | Port the llama.cpp K-quants GEMM/GEMV kernels (`ggml-cuda`'s `mm` kernels for `Q6_K`). No community Python package for Q6_K GEMM. |
| Norms (RMSNorm) | **buy** | torch (fused RMSNorm). The scaffold uses torch for norms. |
| Sampling | **buy** | torch (the scaffold's sampler). |
| Constrained decoding | **buy** | xgrammar or outlines (a community grammar-based sampler). The consumer is agent → the B5 smoke needs one tool-call round trip. |
| KV cache (main, q8_0) | **port** | Port the llama.cpp q8_0 KV quant/dequant (`ggml-cuda`'s k-quants). Matches the origin's `--cache-type-k q8_0 --cache-type-v q8_0`. (Alternative: FlashInfer fp8 KV, 12.5% smaller, but the capacity math and the origin use q8_0.) |
| KV cache (draft, bf16) | **buy** | torch (the draft KV is bf16, no quantization). |
| GDN recurrent state | **port** | Port the llama.cpp GDN state management (the delta-rule update). |
| Vision (mmproj, Qwen3-VL ViT) | **port** | Port the Qwen3-VL ViT + merger from the llama.cpp `clip` model (`src/models/clip.cpp` + the ViT kernels). No community Python package for Qwen3-VL ViT in the scaffold. |
| Speculative decoding (dflash) | **port** | Port the dflash draft-verify scheduler from the llama.cpp `--spec-type draft-dflash` path. The scaffold lacks speculative decoding. |
| Layer split (distributed mode) | **build** | New. The scaffold only has TP (NCCL all-reduce per layer). The profile needs a non-uniform layer split (activation handoff only, no per-layer all-reduce). See §8. |

**Hand-rolled**: none. Every component is either a community implementation (buy) or a
port of a reference-engine kernel (port). The reference oracle (B4) is hand-written f64
by design (the correctness reference, not a performance component).

## 7. Performance techniques (performance-design.md catalog, scenario-gated)

| Technique | In scope? | Rationale (cites SCENARIO.md) |
|---|---|---|
| CUDA graphs (decode) | **yes** | Latency-first; decode is launch-bound. The profile demands decode ≥ 20 tok/s. |
| Operator fusion (SwiGLU, RMSNorm) | **yes** | Standard; the scaffold fuses SwiGLU. |
| Chunked prefill (ubatch 1024) | **yes** | The load has 60k-token inputs; TTFT(60k) < 30 s. |
| Speculative decoding (dflash, n_max 6) | **yes** | The user confirmed it in scope; the origin uses it; the capacity math includes the draft. |
| Quantization (Q6_K weights, q8_0 KV) | **yes** | The capacity check (B1) requires it (bf16 dequant does not fit). |
| Tensor parallelism (TP) | **no** | The profile uses a non-uniform layer split, not TP. TP=2 over PCIe would be ~128 all-reduces/token, eating into the 20 tok/s budget. |
| RadixAttention (prefix caching) | **no** | Only pays with repeated prefixes. The load is concurrency 1, single user, no repeated prefixes. |
| Continuous batching | **no** | Concurrency 1; the profile does not demand it. |

## 8. Multi-GPU: non-uniform layer split

The scaffold's only distributed mode is TP (NCCL all-reduce per layer). Over PCIe (no
NVLink), TP=2 means ~128 all-reduces per token (one per layer), which would add ~6–8 ms
per token and eat into the 20 tok/s budget. The profile instead uses a **non-uniform
layer split**: the 64 main-model layers are partitioned across the 2 GPUs, and the
activation tensor is handed off between GPUs at the split boundary (no per-layer
all-reduce).

- **Split** (B2 design; B3 implements): the 4070 Ti Super (15.99 GiB, rank 0 / cuda:0)
  hosts the first 26 layers (0–25) + the embedding; the 2080 Ti (21.32 GiB usable,
  rank 1 / cuda:1) hosts the remaining 38 layers (26–63) + the lm_head + the draft
  model + the mmproj. The 2080 Ti gets more layers because it has more VRAM and also
  hosts the draft + mmproj. (Measured GPU numbering: GPU0 = 4070 Ti Super, GPU1 =
  2080 Ti; `layer_split = [26, 38]`, `draft_device = 1`.)
- **KV split**: the 16 full-attn layers' KV caches follow the layer split. The 4070 Ti
  Super hosts 6 full-attn layers (3,7,11,15,19,23); the 2080 Ti hosts 10 (27,31,…,63).
  KV is bf16 (4 KB/token/layer) until the q8_0 port lands; the pool self-sizes at boot
  to the measured free memory, so the effective context may sit below 100k until then.
  The draft KV (1.91 GiB) is on the 2080 Ti.
- **Per-GPU budget** @ 100k (bf16 KV):
  - 2080 Ti: 12.2 (main weights) + 1.0 (lm_head) + 3.8 (main KV) + 1.92 (draft) + 0.59
    (mmproj) + 1.91 (draft KV) ≈ 21.4 GiB — tight against 21.32 GiB usable; the KV pool
    self-sizes down at boot if needed.
  - 4070 Ti Super: 8.35 (main weights) + 1.0 (embedding) + 2.3 (main KV) ≈ 11.7 GiB +
    activations < 15.99 GiB ✓
- **Activation handoff**: at the split boundary (layer 25 → 26), the activation tensor
  (hidden 5120 × batch × seq) is copied from the 4070 Ti Super to the 2080 Ti over PCIe
  (gloo P2P with CPU staging). This is one copy per forward pass (not per layer), so the
  overhead is ~1.3 ms/token (vs ~6–8 ms/token for TP).
- **No NCCL**: the split moves activations rank-to-rank over the gloo group (handoff,
  token exchange, the MIN all-reduces in the KV sizing). NCCL allreduce is a uniform-TP
  mechanism the split never calls, so the engine skips the pynccl init when
  `uses_layer_split` (it links libnccl, which this box does not have).

## 9. Memory plan (architecture.md: all regions pre-allocated once at startup)

The engine startup prints the `allocation table:` line with these five regions (plus the
GDN state, which is fixed and does not grow with length):

| Region | Size @ 100k | Notes |
|---|---|---|
| weights | 23.07 GiB | main Q6_K 20.57 + draft Q8_0 1.92 + mmproj Q8_0 0.59; split across 2 GPUs |
| kv | 5.34 GiB | main q8_0 3.43 + draft bf16 1.91 @ 100k; grows with context (100k is the max) |
| activation_workspace | 2 GiB | 60k prefill, ubatch 1024; per-GPU in the layer split |
| pinned_host | 0.2 GiB | per-step H2D/D2H buffers |
| graph_pool | 1 GiB | CUDA graph capture pool; per-GPU in the layer split |
| GDN state | 0.023 GiB | 48 layers × 16 groups × 128 × 128 × bf16; fixed, does not grow |
| **total** | **31.63 GiB** | < 37.31 GiB usable (headroom ~5.7 GiB) |

## 10. Riskiest kernel (flag for early testing in B4)

The **GDN recurrent state update** (the delta-rule update in the 48 GDN layers) is the
most novel and error-prone component. It is a port of the llama.cpp GDN kernel, and it
interacts with the causal conv1d and the gated output. The B4 reference oracle must
verify the GDN state update layer-by-layer at bf16/fp32 before the full-model golden
test. If the GDN kernel is wrong, the whole model is wrong (48 of 64 layers depend on it).

## 11. B2 gate

The profile fully determines the design (no open questions): the model, the hardware, the
use case, the load, and the target are all user-supplied; the capacity check (B1) confirms
the model fits at 100k; the build-vs-buy table resolves every component; the performance
catalog is scenario-gated. **B2 gate: passed** (the profile determines the design).
