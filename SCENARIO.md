# Scenario: qwen38-27b

Purpose-built Python inference engine for exactly one scenario: the Qwen3.8-27B
Gated-Delta-Net hybrid VLM on two consumer GPUs, single-user, latency-first,
long-context, with **MTP (multi-token-prediction) speculative decoding**. Every
field is a user-supplied value (B0 intake) or a checkpoint-verified fact
(gguf-dump). Nothing is inferred.

## Model
- family: multimodal (decoder-llm core + Qwen3-VL vision tower)
- name: Qwen3.8-27B-Coder390-EfficientThink — GGUF arch `qwen35`, a Gated-Delta-Net (GDN) hybrid with an embedded MTP draft block
- params: 27B (model label; dense)
- precision: main **Q6_K**; MTP block (blk.64) **Q8_0** (embedded in the main GGUF, not a separate file); mmproj (vision) **Q8_0**
- checkpoint: gguf
  - main:   `~/Models/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2/Qwen3.8-27B-Coder390-EfficientThink-Q6_K-MTP.gguf` (21.58 GiB; **includes the MTP block blk.64**)
  - mmproj: `.../mmproj-Qwen3.8-27B-Q8_0.gguf` (0.59 GiB; arch `clip`, Qwen3-VL ViT 27 layers + `qwen3vl_merger` projector)
  - **(no separate draft file — the MTP draft is the embedded blk.64, consumed by llama.cpp `--spec-type draft-mtp`)**
- draft model: **MTP only** — the embedded blk.64 is the sole draft path (`--spec-type draft-mtp`); dflash is out of scope (dropped from this scenario)
- hidden 5120 / **64 main layers = 48 GDN (linear-attn) + 16 full-attn** (`full_attention_interval = 4`, full-attn at indices 3,7,11,…,63) / heads 24 / kv_heads 4 / head_dim 256
- **MTP block (blk.64)**: 1 extra full-attn decoder layer + a `nextn` head, embedded in the main GGUF. Tensors (all Q8_0 weights unless noted):
  - `attn_q` [5120, 12288] (q + gate, 2 × 24 × 256), `attn_k` [5120, 1024], `attn_v` [5120, 1024], `attn_output` [6144, 5120]
  - `ffn_gate`/`ffn_up` [5120, 17408], `ffn_down` [17408, 5120]
  - `nextn.eh_proj` [10240, 5120] (projects `[e_norm; h_norm]`, 2×5120 → 5120)
  - norms (F32): `attn_norm`, `post_attention_norm`, `attn_q_norm` [256], `attn_k_norm` [256], `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm` (all [5120])
  - the MTP head uses the main model's `output` (lm_head) for logits (no separate `nextn.shared_head_head` in this GGUF)
- GDN: `state_size 128`, `group_count 16`, `inner_size 6144`, `conv_kernel 4`, `time_step_rank 48`
- context (model max) **262144** / vocab **248320** / rope_theta **1e7** (multi-section `[11,11,10,0]`, `dim_count 64`) / activation **SwiGLU** (silu)
- tokenizer: embedded in GGUF (gpt2-style, pre `qwen35`); eos 248046, bos/pad 248044, add_bos = false
- sampling (origin CLI, baked as defaults): temp 1.0, top-p 0.95, top-k 20, min-p 0.0

## Hardware
- device: 2x NVIDIA — **RTX 4070 Ti Super (16 GB, sm_89)** + **RTX 2080 Ti (22 GB, sm_75 Turing)**
- vram: 38 GB total; the display card (2080 Ti) is budgeted at total − 1G (the driver and display framebuffer reserve there) → **37 GiB usable**
- display_gpu: **CUDA1 (2080 Ti)** — the card that outputs the screen image. User declaration (B0). The B1 probe (`scripts/probe_display_gpu.py`) was inconclusive (exit 2 — headless machine: no X screen, no `Display Active`, no connected DRM connector), so the user declaration stands and is recorded here as a user declaration. Engine budget on that card = 22528 − 1024 = **21 GiB**; the 4070 Ti Super's budget is its full total (**16 GiB**)
- ram: 29 GB (Intel i5-13600KF, 20 threads)
- toolkit: CUDA nvcc 13.0 (driver 616.92)
- gpus: 2, **PCIe (no NVLink)** → non-uniform layer split (more on the 2080 Ti), not high-bandwidth tensor parallel

## Use case
- mode: api (OpenAI-compatible HTTP; single user)
- consumer: **agent** (the single user and the tuning agent both connect to the same service; agent consumer unlocks constrained decoding — the B5 smoke includes one tool-call round trip)
- priority: latency
- target: **decode >= 20 tok/s sustained (concurrency 1), maximize beyond; TTFT for a 60k-token input < 30 s**
- dependent consumers: the tuning agent (connects to the same service; cannot tolerate downtime)
- origin command (verbatim, `{port}` is the port placeholder; confirmed by the user):
  ```
  MTMD_BACKEND_DEVICE=CUDA1 llama-server -m /home/hhy/Models/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2/Qwen3.8-27B-Coder390-EfficientThink-Q6_K-MTP.gguf --mmproj /home/hhy/Models/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2/mmproj-Qwen3.8-27B-Q8_0.gguf --device CUDA0,CUDA1 --tensor-split 22,22 --chat-template-kwargs '{"preserve_thinking": true}' --jinja --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-ngl 99 --spec-draft-device CUDA1 --fit off -t 16 --n-gpu-layers 99 --ctx-size 102400 --batch-size 1024 --ubatch-size 1024 --cache-type-k q8_0 --cache-type-v q8_0 --image-min-tokens 1024 --parallel 1 --temp 1 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --cache-ram 9000 --flash-attn on --load-mode none --host 0.0.0.0 --port {port}
  ```

## Load
- input len: typical ~4k / **max 60k**
- output len: typical ~512 / max 4k
- concurrency: **1**
- streaming: yes
- bench suite: **latency-1** (one length, concurrency 1 — the replacement gate; the other two are reported, not gated)

## Capacity   (B1)
Estimates. The engine startup line `allocation table:` is the measurement of the same five regions.
All figures in GiB. Usable VRAM = 37 GiB (38 total − 1G display reserve on the display card, CUDA1).
Per-card budgets: CUDA0 (4070 Ti Super) 16 GiB (full total); CUDA1 (2080 Ti, display) 21 GiB (total − 1G).

- weights: main Q6_K **21.58** (includes the MTP block blk.64) + mmproj Q8_0 0.59 = **22.17 GiB**
- kv (engine max context = **100k**, the scenario's floor; only the **16 full-attn** main layers + the **1 MTP** full-attn layer grow):
  - main (16 full-attn layers, q8_0 K+V, 1.125 B/elem) @ 100k: **3.43 GiB**
  - MTP (1 full-attn layer, q8_0 K+V) @ 100k: **0.21 GiB**
  - kv total @ 100k: **3.64 GiB**
  - (at the model max 256k the kv would be 8.78 + 0.54 = 9.32 GiB and the total would be 34.72 GiB — **now fits**; but the engine is configured at the 100k floor to match the origin's `--ctx-size 102400`, unchanged from the prior scenario)
- activation_workspace (60k prefill, ubatch 1024): **2 GiB**
- pinned_host (per-step H2D/D2H buffers): **0.2 GiB**
- graph_pool (CUDA graph capture pool): **1 GiB**
- GDN recurrent state (48 layers, **fixed**, does not grow with length): **0.023 GiB**
- **fits: YES @ Q6_K, engine max context = 100k.**
  - 100k total = 22.17 (weights) + 3.64 (kv) + 2 (activation) + 0.2 (pinned) + 1 (graph) + 0.023 (GDN) = **29.03 GiB < 37 GiB usable** (headroom ~8.0 GiB).
  - Non-uniform split (B2): 4070 Ti Super ~12 GiB (< its 16 GiB full-total budget), 2080 Ti ~20 GiB (< its 21 GiB = total − 1G display budget); the MTP block + mmproj land on the 2080 Ti, matching the origin's `--spec-draft-device CUDA1` and `MTMD_BACKEND_DEVICE=CUDA1`.
  - **Floor:** 100k is the engine max context. Never below 100k. (256k now fits, but the floor is unchanged.)
- target: **decode >= 20 tok/s sustained (concurrency 1), maximize beyond; TTFT(60k) < 30 s**
