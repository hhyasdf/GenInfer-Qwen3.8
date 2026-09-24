# Scenario: qwen38-27b

Purpose-built Python inference engine for exactly one scenario: the Qwen3.8-27B
Gated-Delta-Net hybrid VLM on two consumer GPUs, single-user, latency-first,
long-context. Every field is a user-supplied value (B0 intake) or a
checkpoint-verified fact (gguf-dump). Nothing is inferred.

## Model
- family: multimodal (decoder-llm core + Qwen3-VL vision tower)
- name: Qwen3.8-27B-EfficientThink-SimPO — GGUF arch `qwen35`, a Gated-Delta-Net (GDN) hybrid
- params: 27B (model label; dense)
- precision: main **Q6_K**; draft (dflash2) **Q8_0**; mmproj (vision) **Q8_0**
- checkpoint: gguf
  - main:   `~/Models/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF/Qwen3.8-27B-EfficientThink-SimPO-Q6_K.gguf` (20.57 GiB)
  - draft:  `.../dflash2-qwen38-27b-Q8_0.gguf` (1.92 GiB; arch `dflash`, 5 layers, heads 32 / kv 8, head_dim 128 — speculative-decode draft)
  - mmproj: `.../mmproj-Qwen3.8-27B-Q8_0.gguf` (0.59 GiB; arch `clip`, Qwen3-VL ViT 27 layers + `qwen3vl_merger` projector)
- hidden 5120 / layers **64 = 48 GDN (linear-attn) + 16 full-attn** (`full_attention_interval = 4`, full-attn at indices 3,7,11,…,63) / heads 24 / kv_heads 4 / head_dim 256
- GDN: `state_size 128`, `group_count 16`, `inner_size 6144`, `conv_kernel 4`, `time_step_rank 48`
- context (model max) **262144** / vocab **248320** / rope_theta **1e7** (multi-section `[11,11,10,0]`, `dim_count 64`) / activation **SwiGLU** (silu)
- tokenizer: embedded in GGUF (gpt2-style, pre `qwen35`); eos 248046, bos/pad 248044, add_bos = false
- sampling (origin CLI, baked as defaults): temp 0.6, top-p 0.95, top-k 20, min-p 0.0

## Hardware
- device: 2x NVIDIA — **RTX 4070 Ti Super (16 GB, sm_89)** + **RTX 2080 Ti (22 GB, sm_86)**
- vram: 38 GB total; reserve 700 MB on the 2080 Ti for the display → **37.31 GiB usable**
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
  llama-server -m /home/hhy/Models/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF/Qwen3.8-27B-EfficientThink-SimPO-Q6_K.gguf -md /home/hhy/Models/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF/dflash2-qwen38-27b-Q8_0.gguf --mmproj /home/hhy/Models/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF/mmproj-Qwen3.8-27B-Q8_0.gguf --device CUDA0,CUDA1 --tensor-split 22,22 --chat-template-kwargs {"preserve_thinking": true, "reasoning_effort": "medium"} --jinja --spec-type draft-dflash --spec-draft-n-max 6 --spec-draft-ngl 99 --spec-draft-device CUDA1 --fit off -t 16 --n-gpu-layers 99 --ctx-size 102400 --batch-size 1024 --ubatch-size 1024 --cache-type-k q8_0 --cache-type-v q8_0 --image-min-tokens 1024 --parallel 1 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --cache-ram 12000 --flash-attn on --load-mode none --host 0.0.0.0 --port {port}
  ```

## Load
- input len: typical ~4k / **max 60k**
- output len: typical ~512 / max 4k
- concurrency: **1**
- streaming: yes
- bench suite: **latency-1** (one length, concurrency 1 — the replacement gate; the other two are reported, not gated)

## Capacity   (B1)
Estimates. The engine startup line `allocation table:` is the measurement of the same five regions.
All figures in GiB. Usable VRAM = 37.31 GiB.

- weights: main Q6_K 20.57 + draft Q8_0 1.92 + mmproj Q8_0 0.59 = **23.07 GiB**
- kv (engine max context = **100k**, the scenario's floor; only the **16 full-attn** main layers + the **5** draft layers grow):
  - main (16 full-attn layers, q8_0 K+V, 1.125 B/elem) @ 100k: **3.43 GiB**
  - draft (5 layers, bf16 K+V, 2 B/elem) @ 100k: **1.91 GiB**
  - kv total @ 100k: **5.34 GiB**
  - (at the model max 256k the kv would be 9.00 + 5.00 = 14.00 GiB and the total would be 40.29 GiB — **does not fit**; hence the engine is configured at the 100k floor, not 256k)
- activation_workspace (60k prefill, ubatch 1024): **2 GiB**
- pinned_host (per-step H2D/D2H buffers): **0.2 GiB**
- graph_pool (CUDA graph capture pool): **1 GiB**
- GDN recurrent state (48 layers, **fixed**, does not grow with length): **0.023 GiB**
- **fits: YES @ Q6_K, engine max context = 100k.**
  - 100k total = 23.07 (weights) + 5.34 (kv) + 2 (activation) + 0.2 (pinned) + 1 (graph) + 0.023 (GDN) = **31.63 GiB < 37.31 GiB usable** (headroom ~5.7 GiB).
  - Non-uniform split (B2): 4070 Ti Super ~12 GiB, 2080 Ti ~20 GiB (both fit; the draft + mmproj land on the 2080 Ti).
  - **Floor:** 100k is the engine max context. Never below 100k. (256k does not fit once the draft kv is counted.)
- target: **decode >= 20 tok/s sustained (concurrency 1), maximize beyond; TTFT(60k) < 30 s**
