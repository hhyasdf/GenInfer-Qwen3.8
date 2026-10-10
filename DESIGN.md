# DESIGN — qwen38-27b engine

Design decisions for the Qwen3.8-27B GDN-hybrid VLM engine. Every decision cites a
field in `SCENARIO.md`. The scaffold (mini-sglang, CUDA-only) is the starting point;
this document records the deltas the profile demands.

## 1. Overview

- **Model**: Qwen3.8-27B Gated-Delta-Net hybrid VLM (GGUF arch `qwen35`). 64 main layers =
  48 GDN (linear-attention) + 16 full-attention (at indices 3,7,11,…,63), plus **1 embedded
  MTP block (blk.64)** — a full-attention decoder layer + a `nextn` head that performs
  multi-token-prediction speculative decoding. Hidden 5120, heads 24 / kv 4 / head_dim 256.
  Context 262144 (model max), vocab 248320.
- **Precision**: main Q6_K, MTP block (blk.64) Q8_0 (embedded in the main GGUF), mmproj
  Q8_0 (GGUF-quantized, kept quantized — no dequantize-to-bf16; the 54 GB bf16 dequant does
  not fit 37 GiB).
- **Hardware**: 2× PCIe GPU (4070 Ti Super sm_89 16 GB + 2080 Ti **sm_75** Turing 22 GB,
  the **Display GPU** — user declaration, B0; the B1 probe was inconclusive (headless), so
  the declaration stands). No NVLink. Per-card budgets: 4070 Ti Super 16 GiB (full total),
  2080 Ti 21 GiB (total − 1G display reserve) → 37 GiB usable. Non-uniform layer split
  (more on the 2080 Ti).
- **Use case**: OpenAI-compatible HTTP API, single user, consumer = **agent** (unlocks
  constrained decoding), latency-first.
- **Target**: decode ≥ 20 tok/s sustained @ concurrency 1; TTFT(60k) < 30 s.
- **Engine max context**: **100k** (the scenario's floor, matching the origin's
  `--ctx-size 102400`). 256k now fits (34.72 GiB < 37 GiB) because the MTP block's KV
  cache (1 full-attn layer, 0.54 GiB @ 256k) is far smaller than the dflash draft's
  (5 layers, 5.00 GiB @ 256k), but the floor is unchanged.

## 2. Backend

- **CUDA** (the scaffold is CUDA-only; the profile is 2× NVIDIA GPU).
- **Attention backend seam** (`--attn`): `triton_fa` (Triton FA, vLLM pattern) for the
  **16 full-attn main layers** and the **1 MTP full-attn layer** (B3: the scaffold's
  `fi`/FlashInfer default is rejected on the back rank — see §6). The **48 GDN layers** do
  not use the seam — they use a ported llama.cpp GDN (delta-net) CUDA kernel (see §6,
  build-vs-buy). The seam choice does not touch the GDN path.
- **Heterogeneous sm_75 + sm_89**: GPU0 = 4070 Ti Super (sm_89), GPU1 = 2080 Ti
  (**sm_75 Turing**, compute_cap 7.5 — not sm_86 as B2 assumed). The kernels must compile
  for both. This is why the attention backend is Triton FA: every flash-style fused
  attention option is rejected on sm_75 (FlashInfer FA2 head_dim-256 prefill core-dumps;
  sgl-kernel FA2/FA3 dispatch rejects "Ampere or newer"; cuDNN frontend SDPA requires
  SM80+; torch SDPA rejects GQA and unequal-length in this build). Triton supports sm_75
  and the ported llama.cpp GDN kernel runs on both. Verified at B3 (boot green on both
  ranks; the 2-GPU layer-split path is new).

## 3. Model families (deltas from the scaffold)

The scaffold supports llama, mistral, qwen2, qwen3, qwen3_moe. This profile needs three
new families (per `references/models/llm-decoder.md`: "A family that is not in the table
is unsupported until DESIGN.md records its deltas and the class is registered"):

### 3.1 `qwen35` (main model) — GDN hybrid
- **Deltas from qwen3**: 48 of the 64 main layers are GDN (linear-attention) instead of
  full-attention. The GDN layer has: a causal conv1d (kernel 4), a delta-rule recurrent
  state (state_size 128, group_count 16, time_step_rank 48, inner_size 6144), and a
  gated output. The 16 full-attn layers are standard multi-head attention with QK-norm
  (Qwen3-style) and head_dim 256.
- **RoPE**: multi-section `[11,11,10,0]`, dim_count 64, base 1e7 (applied to the
  full-attn layers only; the GDN layers use the recurrent state, not RoPE).
- **Activation**: SwiGLU (silu × gate), ffn 17408.
- **Tokenizer**: GGUF-embedded gpt2 BPE (no HF tokenizer.json). Port the llama.cpp vocab
  loader (see §6).
- **Hidden-state export for MTP**: the main model must expose its last hidden state `h`
  (before the final output norm) when MTP speculative decoding is enabled. This is the
  `t_h_nextn` input to the MTP block (see §3.2). The main model's forward pass returns both
  the logits and the hidden state `h`.
- **Class**: `Qwen35ForCausalLM` (new). Registered in the model registry.

### 3.2 `mtp` (draft model) — embedded MTP block (blk.64)
- **Deltas**: a single full-attention decoder layer (blk.64) + a `nextn` head, embedded in
  the main GGUF (not a separate file). Tensors: `attn_q` [5120, 12288] (q + gate),
  `attn_k`/`attn_v` [5120, 1024], `attn_output` [6144, 5120], `ffn_gate`/`ffn_up`
  [5120, 17408], `ffn_down` [17408, 5120], `nextn.eh_proj` [10240, 5120], and F32 norms
  (`attn_norm`, `post_attention_norm`, `attn_q_norm` [256], `attn_k_norm` [256],
  `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm`). All weights Q8_0.
- **Algorithm** (llama.cpp `graph_mtp`, `--spec-type draft-mtp`): given the main model's
  hidden state `h` (before output norm) and the last token's embedding `e`:
  1. `h_norm = RMSNorm(h, nextn.hnorm)`, `e_norm = RMSNorm(e, nextn.enorm)`.
  2. `eh = concat([e_norm; h_norm])` → [10240, n_tokens].
  3. `cur = eh_proj(eh)` → [5120, n_tokens] (residual `inpSA`).
  4. `cur = RMSNorm(cur, attn_norm)`.
  5. `[Q_full, K, V] = build_qkv(cur)`; `Q = RMSNorm(Q, attn_q_norm)`, `K = RMSNorm(K,
     attn_k_norm)`; apply RoPE to Q, K.
  6. `cur = attention(Q, K, V) * sigmoid(gate)` (gated attention); `cur = attn_output(cur)`.
  7. `cur = cur + inpSA`; `ffn_residual = cur`; `cur = RMSNorm(cur, post_attention_norm)`.
  8. `cur = FFN(cur)` (SwiGLU); `cur = cur + ffn_residual`.
  9. `cur = RMSNorm(cur, nextn.shared_head_norm)` → new hidden state `h'` (for chaining).
  10. `logits = lm_head(cur)` (the main model's `output` weight; no separate
      `nextn.shared_head_head` in this GGUF) → draft token.
- **Role**: generates up to `n_max = 4` draft tokens per step by chaining the MTP block on
  its own hidden state (`h'` becomes the next step's `h`); the main model verifies them in
  one forward pass (see §7, speculative decoding). The MTP block's KV cache is for the full
  context (1 full-attn layer, q8_0).
- **Class**: `MtpForCausalLM` (new, replaces `DflashForCausalLM`). Registered in the model
  registry under the `mtp` arch. The `dflash` arch is dropped from the registry.

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
| Attention (full-attn, 16 main + 1 MTP layers) | **build** | Triton FA (vLLM pattern), `triton_fa` backend. B3 rejected every community flash-style option on the back rank (GPU1 = sm_75 Turing): FlashInfer FA2 head_dim-256 prefill core-dumps (5 variants, identical crash); sgl-kernel FA2/FA3 dispatch rejects "Ampere or newer" for cc=(7,5); cuDNN frontend SDPA requires SM80+; torch SDPA rejects GQA and unequal-length in this build. Triton supports sm_75; the kernel is verified against an fp32 math reference on both GPUs (rel err ~2e-3) and boot-confirmed. The MTP block's full-attn layer uses the same Triton FA kernel. |
| Attention (GDN, 48 layers) | **port** | Port the llama.cpp GDN (delta-net) CUDA kernel (`src/models/delta-net-base.cpp` + the GDN kernels in `ggml-cuda`). No community Python package for GDN. |
| GEMM (Q6_K weights) | **port** | Port the llama.cpp K-quants GEMM/GEMV kernels (`ggml-cuda`'s `mm` kernels for `Q6_K`). No community Python package for Q6_K GEMM. |
| GEMM (MTP block, Q8_0 weights) | **port** | The MTP block's weights are Q8_0; use the llama.cpp Q8_0 GEMM/GEMV kernels (`ggml-cuda`'s `mm` kernels for `Q8_0`). |
| Norms (RMSNorm) | **buy** | torch (fused RMSNorm). The scaffold uses torch for norms. |
| Sampling | **buy** | torch (the scaffold's sampler). |
| Constrained decoding | **buy** | xgrammar or outlines (a community grammar-based sampler). The consumer is agent → the B5 smoke needs one tool-call round trip. |
| KV cache (main, q8_0) | **port** | Port the llama.cpp q8_0 KV quant/dequant (`ggml-cuda`'s k-quants). Matches the origin's `--cache-type-k q8_0 --cache-type-v q8_0`. |
| KV cache (MTP, q8_0) | **port** | The MTP block's KV cache is q8_0 (1 full-attn layer), same quantization as the main model's full-attn KV. Port the same llama.cpp q8_0 KV kernels. |
| GDN recurrent state | **port** | Port the llama.cpp GDN state management (the delta-rule update). |
| Vision (mmproj, Qwen3-VL ViT) | **port** | Port the Qwen3-VL ViT + merger from the llama.cpp `clip` model (`src/models/clip.cpp` + the ViT kernels). No community Python package for Qwen3-VL ViT in the scaffold. |
| Speculative decoding (MTP) | **port** | Port the MTP draft-verify scheduler from the llama.cpp `--spec-type draft-mtp` path (`src/models/qwen35.cpp` `graph_mtp`). The scaffold lacks speculative decoding. The MTP block is a single embedded decoder layer (blk.64), not a separate model — the scheduler chains it on its own hidden state and verifies with the main model. |
| Layer split (distributed mode) | **build** | New. The scaffold only has TP (NCCL all-reduce per layer). The profile needs a non-uniform layer split (activation handoff only, no per-layer all-reduce). See §8. |

**Hand-rolled**: the Triton FA kernel (full-attn, including the MTP block's layer) is the
one build — justified because every community flash-style backend rejects sm_75 (the back
rank), so no suitable community implementation exists for the target hardware (evidence
above). Every other component is either a community implementation (buy) or a port of a
reference-engine kernel (port). The reference oracle (B4) is hand-written f64 by design
(the correctness reference, not a performance component).

## 7. Performance techniques (performance-design.md catalog, scenario-gated)

| Technique | In scope? | Rationale (cites SCENARIO.md) |
|---|---|---|
| CUDA graphs (decode) | **yes** | Latency-first; decode is launch-bound. The profile demands decode ≥ 20 tok/s. |
| Operator fusion (SwiGLU, RMSNorm) | **yes** | Standard; the scaffold fuses SwiGLU. |
| Chunked prefill (ubatch 1024) | **yes** | The load has 60k-token inputs; TTFT(60k) < 30 s. |
| Speculative decoding (MTP, n_max 4) | **yes** | The user confirmed it in scope (replaces dflash); the origin uses `--spec-type draft-mtp --spec-draft-n-max 4`; the capacity math includes the MTP block's KV. |
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

- **Split** (B2 design; B3 implements): the 4070 Ti Super (15.99 GiB full total, rank 0 /
  cuda:0) hosts the first 26 main layers (0–25) + the embedding; the 2080 Ti (21 GiB engine
  budget — the **Display GPU**, capped at total − 1G, rank 1 / cuda:1) hosts the remaining
  38 main layers (26–63) + the lm_head + the **MTP
  block (blk.64)** + the mmproj. The 2080 Ti gets more layers because it has more VRAM and
  also hosts the MTP block + mmproj (matching the origin's `--spec-draft-device CUDA1` and
  `MTMD_BACKEND_DEVICE=CUDA1`). (Measured GPU numbering: GPU0 = 4070 Ti Super, GPU1 =
  2080 Ti; `layer_split = [26, 38]`, `draft_device = 1`.)
- **KV split**: the 16 full-attn main layers' KV caches follow the layer split. The 4070 Ti
  Super hosts 6 full-attn layers (3,7,11,15,19,23); the 2080 Ti hosts 10 (27,31,…,63).
  The **MTP block's KV cache** (1 full-attn layer, q8_0) is on the 2080 Ti (with the MTP
  block). KV is q8_0 (1.125 B/elem) matching the origin; the pool self-sizes at boot to the
  measured free memory, so the effective context may sit below 100k until then.
- **Per-GPU budget** @ 100k (q8_0 KV) — per-card budgets from B1 (the display card, the
  2080 Ti, is total − 1G):
  - 2080 Ti (display): 12.2 (main weights) + 1.0 (lm_head) + 0.9 (MTP block weights) + 3.8
    (main KV) + 0.21 (MTP KV) + 0.59 (mmproj) ≈ 18.7 GiB < 21 GiB (total − 1G) ✓
  - 4070 Ti Super (full total): 8.35 (main weights) + 1.0 (embedding) + 2.3 (main KV)
    ≈ 11.7 GiB + activations < 15.99 GiB ✓
- **Activation handoff**: at the split boundary (layer 25 → 26), the activation tensor
  (hidden 5120 × batch × seq) is copied from the 4070 Ti Super to the 2080 Ti over PCIe
  (gloo P2P with CPU staging). This is one copy per forward pass (not per layer), so the
  overhead is ~1.3 ms/token (vs ~6–8 ms/token for TP). The MTP block's hidden state `h`
  (before output norm) is also handed to the MTP block on the 2080 Ti (same GPU as the
  lm_head, so no extra cross-GPU copy for the MTP path).
- **No NCCL**: the split moves activations rank-to-rank over the gloo group (handoff,
  token exchange, the MIN all-reduces in the KV sizing). NCCL allreduce is a uniform-TP
  mechanism the split never calls, so the engine skips the pynccl init when
  `uses_layer_split` (it links libnccl, which this box does not have).

## 9. Memory plan (architecture.md: all regions pre-allocated once at startup)

The engine startup prints the `allocation table:` line with these five regions (plus the
GDN state, which is fixed and does not grow with length):

| Region | Size @ 100k | Notes |
|---|---|---|
| weights | 22.17 GiB | main Q6_K 21.58 (includes MTP block blk.64) + mmproj Q8_0 0.59; split across 2 GPUs |
| kv | 3.64 GiB | main q8_0 3.43 (16 full-attn layers) + MTP q8_0 0.21 (1 full-attn layer) @ 100k; grows with context (100k is the max) |
| activation_workspace | 2 GiB | 60k prefill, ubatch 1024; per-GPU in the layer split |
| pinned_host | 0.2 GiB | per-step H2D/D2H buffers |
| graph_pool | 1 GiB | CUDA graph capture pool; per-GPU in the layer split |
| GDN state | 0.023 GiB | 48 layers × 16 groups × 128 × 128 × bf16; fixed, does not grow |
| **total** | **29.03 GiB** | < 37 GiB usable (38 total − 1G display reserve on the 2080 Ti; headroom ~8 GiB) |

## 10. Riskiest kernel (flag for early testing in B4)

The **GDN recurrent state update** (the delta-rule update in the 48 GDN layers) is the
most novel and error-prone component. It is a port of the llama.cpp GDN kernel, and it
interacts with the causal conv1d and the gated output. The B4 reference oracle must
verify the GDN state update layer-by-layer at bf16/fp32 before the full-model golden
test. If the GDN kernel is wrong, the whole model is wrong (48 of 64 layers depend on it).

The **MTP block's hidden-state chaining** is the second riskiest component: the MTP block
must correctly consume the main model's hidden state `h` (before output norm), and the
chained hidden states (`h'`) must be correct across the `n_max = 4` draft steps. A wrong
`h` or a wrong `h'` produces wrong draft tokens (the main model's verification will reject
them, but the acceptance rate collapses). The B4 reference oracle must verify the MTP
block's hidden-state export and chaining layer-by-layer at bf16/fp32.

## 11. B2 gate

The profile fully determines the design (no open questions): the model, the hardware, the
use case, the load, and the target are all user-supplied; the capacity check (B1) confirms
the model fits at 100k (with more headroom than the dflash scenario); the build-vs-buy
table resolves every component; the performance catalog is scenario-gated. **B2 gate:
passed** (the profile determines the design).
