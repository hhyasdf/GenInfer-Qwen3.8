# Phase
track: build
phase: B4
status: ready
note: B0/B1/B2/B3 done; B4 = correctness: bake golden logits from the independent reference oracle, prove the engine matches layer-by-layer at bf16/fp32, then a short greedy. Blockers to clear first: bug B (ssm_conv1d garbage/zero weights in the GGUF), the logits[:batch.size] prefill-sampling question, and golden/logits.json is still the toy model's

## History
- B0 done — model + checkpoint named; origin command confirmed verbatim (llama-server :8080, pid 4152); consumer=agent; bench=latency-1; vision in scope; spec-decode in scope; GGUF kept Q6_K (user ruling)
- B1 done — SCENARIO.md complete format; capacity check corrected (draft KV was missing): 256k does not fit (40.29 GiB > 37.31), 100k fits (31.63 GiB); engine max context = 100k (scenario floor)
- B2 done — DESIGN.md written; profile fully determines the design (B2 gate passed); 3 new model families (qwen35 GDN hybrid, dflash draft, clip mmproj); build-vs-buy table resolves every component; non-uniform layer split (not TP); GDN recurrent state update flagged as riskiest kernel
- B3 done — real Qwen3.8-27B GGUF loads and serves on the layer-split (2-rank gloo) engine; 1-token chat completion answers (200 OK); startup log prints `stack:` and `allocation table:`; tensor inventory matches config. Attention backend = **Triton FA** (FlashInfer FA2 head_dim-256 prefill crashes on sm_75; sgl-kernel/cuDNN/torch-SDPA all reject sm_75 — see DESIGN.md build-vs-buy). Layer-split handoff fixed (back-rank recv sized by token count `input_ids.shape[0]`, not `batch.size` req count). `gates --phase B3` = ok. Known B4 blockers: bug B (ssm_conv1d garbage/zero in the GGUF), the `logits[:batch.size]` prefill-sampling question, golden/logits.json still the toy model's
- Scenario regenerated 2026-10-09 — origin command re-confirmed verbatim by the user; the only diff vs the prior confirmed command is `--temp 0.6` → `--temp 1` (sampling default now temp 1.0); draft model = **MTP only** (dflash dropped — no dflash arch, code, or file remains); model path and every other scenario field unchanged

## B0 intake (user-supplied, from SCENARIO.md)
- model: Qwen3.8-27B GDN hybrid, GGUF arch `qwen35`, Q6_K 21.58 GiB (**includes the embedded MTP block blk.64**) + mmproj Q8_0 0.59 GiB, `~/Models/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2/`
- draft model: MTP only (embedded blk.64, `--spec-type draft-mtp`); dflash is out of scope
- hardware: 2× PCIe GPU — 4070 Ti Super (sm_89, 16376 MiB) + 2080 Ti (sm_75, 22528 MiB, 700 MB display reserve); no NVLink; usable ~37.31 GiB; RAM 29 GB
- use case: streaming API, single user + tuning agent (same service); latency priority
- load: input 4k–60k, output ≤ 4k, concurrency 1, streaming yes
- target: decode ≥ 20 tok/s @ concurrency 1; TTFT(60k) < 30 s
- constraints: offline; origin llama-server on :8080 must not stop outside supervisor transactions

## Endpoint probe (references/endpoint-probe.md)
- env whitelist (OPENAI_BASE_URL / OPENAI_API_BASE / LLM_BASE_URL / LOCAL_MODEL_URL / OLLAMA_HOST): none set
- listener: llama-server on :8080 (the agent port); the engine takes the port only through `engine-verify`
- confirmed origin command: **confirmed verbatim by the user (2026-10-09)** — recorded in SCENARIO.md / `[origin].command` with the `{port}` placeholder; the `MTMD_BACKEND_DEVICE=CUDA1` prefix (vision backend on CUDA1) is retained as in the previously confirmed line

## Checkpoint facts (machine-verified from GGUF metadata)
- main Q6_K, arch `qwen35`, 21.58 GiB: 64 main blocks — 48 GDN + 16 full-attn at indices 3,7,…,63 (`full_attention_interval` 4); hidden 5120; ffn 17408; 24 heads / 4 kv heads; head_dim 256; rope base 1e7, sections [11,11,10,0], dim 64; ssm {conv 4, state 128, group 16, dt_rank 48, inner 6144}; ctx 262144; vocab 248320; gpt2 tokenizer (eos 248046, bos/pad 248044, add_bos false)
- embedded MTP block blk.64 (Q8_0 weights, F32 norms): 1 full-attn decoder layer + `nextn` head; `attn_q` [5120,12288] (q+gate), `nextn.eh_proj` [10240,5120]; shares the main model's `output` (lm_head) for logits
- mmproj, arch `clip`, Q8_0, 0.59 GiB: 27 ViT layers + patch_embd / position_embd / post_ln + `qwen3vl_merger` projector
- sampling (origin CLI): temp 1.0, top-p 0.95, top-k 20, min-p 0.0
- local llama.cpp commit 718f7b417 (b10927) implements `qwen35` natively (`src/models/qwen35.cpp`, `delta-net-base.cpp`); **no GGUF→HF converter in the tree**
