# Phase
track: build
phase: B3
status: in_progress
note: B0/B1/B2 done; B3 = copy scaffold, fill config.toml, add qwen35/dflash/clip model classes, port K-quants GEMM + GDN kernels + q8_0 KV + vocab loader, add layer-split distributed mode + speculative decoding + constrained decoding

## History
- B0 done — model + checkpoint named; origin command confirmed verbatim (llama-server :8080, pid 4152); consumer=agent; bench=latency-1; vision in scope; spec-decode in scope; GGUF kept Q6_K (user ruling)
- B1 done — SCENARIO.md complete format; capacity check corrected (draft KV was missing): 256k does not fit (40.29 GiB > 37.31), 100k fits (31.63 GiB); engine max context = 100k (scenario floor)
- B2 done — DESIGN.md written; profile fully determines the design (B2 gate passed); 3 new model families (qwen35 GDN hybrid, dflash draft, clip mmproj); build-vs-buy table resolves every component; non-uniform layer split (not TP); GDN recurrent state update flagged as riskiest kernel

## B0 intake (user-supplied, from SCENARIO.md)
- model: Qwen3.8-27B GDN hybrid, GGUF arch `qwen35`, Q6_K 23.6 GB + dflash2 draft Q8_0 2.0 GB + mmproj Q8_0 0.6 GB, `~/Models/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF/`
- hardware: 2× PCIe GPU — 4070 Ti Super (sm_89, 16376 MiB) + 2080 Ti (sm_86, 22528 MiB, 700 MB display reserve); no NVLink; usable ~37.3 GB; RAM 29 GB
- use case: streaming API, single user + tuning agent (same service); latency priority
- load: input 4k–60k, output 1k–8k, concurrency 1, streaming yes
- target: decode ≥ 20 tok/s @ concurrency 1; TTFT(60k) < 30 s
- constraints: offline; origin llama-server on :8080 must not stop outside supervisor transactions

## Endpoint probe (references/endpoint-probe.md)
- env whitelist (OPENAI_BASE_URL / OPENAI_API_BASE / LLM_BASE_URL / LOCAL_MODEL_URL / OLLAMA_HOST): none set
- listener: pid 4152, `llama-server … --host 0.0.0.0 --port 8080` (full cmdline recorded in session)
- confirmed origin command: **pending** (user to confirm verbatim with `{port}`)
- leftover: scaffold toy engine (pid 6287/6319) on :8081 — harmless, holds a little VRAM
- machine vs scenario: driver 616.92 (scenario says 615.71.08); nvcc 13.0 (scenario says CUDA 13.4)

## Checkpoint facts (machine-verified from GGUF metadata)
- main Q6_K, arch `qwen35`: 64 blocks — 48 GDN + 16 full-attn at indices 3,7,…,63 (`full_attention_interval` 4); hidden 5120; ffn 17408; 24 heads / 4 kv heads; head_dim 256; rope base 1e7, sections [11,11,10,0], dim 64; ssm {conv 4, state 128, group 16, dt_rank 48, inner 6144}; ctx 262144; vocab 248320; gpt2 tokenizer (eos 248046, bos/pad 248044, add_bos false); no MTP tensors
- draft, arch `dflash`: 5 blocks; hidden 5120; ffn 17408; 32 heads / 8 kv; head_dim 128; ctx 262144; conv layers inside blocks
- mmproj, arch `clip`: 27 ViT layers + patch_embd / position_embd / post_ln + mm merger
- sampling (origin CLI): temp 0.6, top-p 0.95, top-k 20, min-p 0.0
- local llama.cpp commit 718f7b417 (b10927) implements `qwen35` natively (`src/models/qwen35.cpp`, `delta-net-base.cpp`); **no GGUF→HF converter in the tree**
