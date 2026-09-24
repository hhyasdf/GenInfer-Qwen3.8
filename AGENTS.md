# AGENTS.md

A gen-inference engine built for one scenario. Read this before changing anything.

> **Scaffold note:** this is the template for the engine's `AGENTS.md`. The
> skill fills every `<placeholder>` from `SCENARIO.md`, `DESIGN.md`, and the
> B1 performance target (filled in B3). The invariants, verification commands,
> key files, and tuning protocol below are fixed — they do not change per
> scenario.

## What this engine is

A purpose-built Python inference engine for one scenario: Qwen3.8-27B
GDN-hybrid VLM (GGUF Q6_K) on 2× PCIe GPU (4070 Ti Super + 2080 Ti), serving
an OpenAI-compatible streaming HTTP API for a single agent under a 60k-input
latency-first load.

The engine is `minisgl` — a Python/torch inference engine copied from
[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang) (MIT),
installed editable into `.venv`. It is a FastAPI/uvicorn frontend plus one
scheduler worker per GPU plus tokenizer/detokenizer workers, all children of
the frontend process. A worker watchdog in the frontend exits the whole
engine when any worker dies (it logs `Backend worker <name> died
(exitcode=N)`), so a crashed scheduler never leaves a wedged API server
behind. The target backend is CUDA (mini-sglang is CUDA-only);
the correctness oracle is independent of the engine (`reference.py`). The
toy uses float64. A real model uses one HF layer at bf16 or fp32, then a
short greedy (`references/backends/cpu.md`).

## Scenario (key facts)

- **Model**: Qwen3.8-27B Gated-Delta-Net hybrid VLM (GGUF arch `qwen35`),
  64 layers = 48 GDN + 16 full-attn, hidden 5120, ctx 262144 (engine max
  100k), vocab 248320. Weights Q6_K (main) + Q8_0 (dflash2 draft, mmproj
  Qwen3-VL ViT). GGUF-only checkpoint (no HF safetensors, no tokenizer.json).
- **Hardware**: 2× PCIe GPU (no NVLink): 4070 Ti Super 16 GB (sm_89) +
  2080 Ti 22 GB (sm_86, 700 MB display reserve). 37.31 GiB usable. 29 GB
  RAM. Toolkit: CUDA 13.0, driver 616.92.
- **Use case**: OpenAI-compatible streaming HTTP API, single user,
  latency-first. The agent port (8080) already serves the origin
  llama-server.
- **Load**: input up to 60k tokens, output up to 4k tokens, concurrency 1,
  streaming.
- **Consumer**: agent (unlocks constrained decoding; the B5 smoke includes
  one tool-call round trip).
- **Dependent consumers**: yes — the agent port (8080) serves the origin
  llama-server (pid 4152). The supervisor owns any stop of that port.

## Endpoint evidence

Filled from `references/endpoint-probe.md`. The probe does not write
`[origin].command`.

- env: none (no env var names the model or a port)
- listen: 0.0.0.0:8080, pid 4152, llama-server (the origin service)
- confirmed origin command: verbatim (the llama-server line in
  `config.toml` `[origin].command`, confirmed by the user in B0; `{port}`
  is the supervisor's placeholder)

Full profile: `SCENARIO.md`.

## Performance target

Decode ≥ 20 tok/s sustained at concurrency 1; TTFT(60k) < 30 s. The bench
suite is `latency-1` (one length at concurrency 1), prompt_tokens 60000,
new_tokens 4096, runs 5, warmup 2. The engine must beat the origin
llama-server baseline on this suite to bind the agent port.

## Design (key decisions)

- **Modules**: the minisgl package: server (frontend + scheduler +
  tokenizer/detokenizer workers), model (the qwen35 GDN-hybrid decoder + the
  dflash draft + the Qwen3-VL vision tower), attention (the fi backend seam
  for the 16 full-attn layers; the 48 GDN layers use the ported llama.cpp
  GDN kernel), cache (naive PagedAttention, q8_0 KV), sampler (torch +
  xgrammar constrained decoding), distributed (the non-uniform layer split).
- **Data flow**: tokens → scheduler → model.forward (48 GDN + 16 full-attn
  layers, split across 2 GPUs) → sampler → tokens. The draft model generates
  up to 6 draft tokens per step; the main model verifies them (speculative
  decoding).
- **Memory plan**: all regions pre-allocated once at startup — weights
  23.07 GiB (main Q6_K 20.57 + draft Q8_0 1.92 + mmproj Q8_0 0.59, split
  across 2 GPUs), KV 5.34 GiB @ 100k (main q8_0 3.43 + draft bf16 1.91),
  activation workspace 2 GiB, pinned host 0.2 GiB, graph pool 1 GiB, GDN
  state 0.023 GiB (fixed). Total 31.63 GiB < 37.31 GiB usable. Never
  reallocated in the serving loop.
- **Batching**: single-stream (concurrency 1, no continuous batching) +
  chunked prefill (ubatch 1024) for the 60k inputs.
- **Backend**: CUDA (mini-sglang). The oracle is the baked golden file
  (HF transformers reference, layer-by-layer at bf16/fp32).
- **Components (build vs buy)**: attention (full-attn) = FlashInfer (fi
  backend, buy); attention (GDN) = ported llama.cpp GDN kernel (port); GEMM
  (Q6_K) = ported llama.cpp K-quants GEMM/GEMV (port); KV (q8_0) = ported
  llama.cpp q8_0 quant/dequant (port); tokenizer = ported llama.cpp vocab
  loader (port); norms/sampling = torch (buy); constrained decoding =
  xgrammar (buy); vision (Qwen3-VL ViT) = ported llama.cpp clip (port);
  speculative decoding (dflash) = ported llama.cpp draft-verify (port);
  layer split = new (build). Full table: `DESIGN.md` §6.

Full design: `DESIGN.md`.

## Invariants (verify before committing)

1. `.venv` builds: `uv venv .venv && uv pip install -e ".[dev]"` (or
   `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`) succeeds.
2. `.venv/bin/python -m pytest tests/` passes.
3. `.venv/bin/python -m minisgl --selftest` passes (determinism + golden).
4. The cmd entry point (`.venv/bin/python -m minisgl`, zero arguments) reaches
   a healthy service: `GET /health`, `GET /v1/models`, and a streamed
   `/v1/chat/completions` all round-trip.

## Verification commands

```bash
uv venv .venv && uv pip install -e ".[dev]"   # build (or python3 -m venv .venv && .venv/bin/pip install -e ".[dev]")
.venv/bin/python -m pytest tests/
.venv/bin/python -m minisgl --selftest
.venv/bin/python -m minisgl   # in one terminal; then:
curl -s localhost:<port>/health
curl -s localhost:<port>/v1/models
curl -sN localhost:<port>/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"models/tiny-llama","messages":[{"role":"user","content":"hello"}],"max_tokens":8}'
```

> **`ninja` on PATH.** The kernel/GPU tests AOT-compile a small C++ extension
> at import time (tvm_ffi shells out to `ninja`), so `pytest tests/` needs
> `ninja` on PATH. The dev extra installs it into the venv — activate the venv
> (`source .venv/bin/activate`) before running the tests, or install `ninja`
> system-wide (`pip install ninja` / `apt install ninja-build`).

## Key files

- `PHASE.md` — the cursor: Build (B0–B5) or Tune (T0–T3), and what is left in the open phase.
- `SCENARIO.md` — the filled scenario profile.
- `DESIGN.md` — the design (modules, data flow, memory plan, build-vs-buy table).
- `TUNING.md` — the tuning log (T0 baseline, T1 one bottleneck, T2 one change).
- `VERIFICATION.md` — gates 1–2 from B5; the performance section from T3.
- `README.md` — the human-facing overview.
- `config.toml` — the baked-in config: performance coefficients (context,
  batching, backend, port, health window) + quality defaults (sampling, model
  path — CLI-overridable at runtime).
- `python/minisgl/` — the engine package (the mini-sglang code tree + the
  contract layer: `admin.py`, `completions.py`, `selftest.py`, `bench.py`).
- `reference.py` — the toy float64 reference; writes
  `golden/logits.json`.
- `golden/` — the trusted reference artifact for `--selftest`.
- `models/tiny-llama/` — the scaffold's toy model (the skill replaces it with
  the real checkpoint).
- `logs/supervisor.log` — the supervisor's log (its lines + the child's
  output); the post-mortem for a dead session. Startup renames the previous
  file to `logs/supervisor.log.prev` and opens a fresh log.

## Tuning protocol

The tuning agent is a client, never a resident. The engine runs under the
skill's supervisor when a transaction is in progress; the tuner writes
`.supervisor-action` and reads HTTP. Tune runs T0 (measure) → T1 (one
bottleneck) → T2 (`engine-verify`) → T1 again until T3 (accept). The engine
stays on the agent port only when that verify shows it serves and beats the
origin baseline on `[bench].suite`. A `native` edit changes the JIT cache
key, which is a hash of `kernel/csrc`. A `dep` edit needs the venv
reinstalled first. The step ring
is on `/admin/stats` and in `logs/verify-report.json`. See `TUNING.md` and
`PHASE.md`.

**GPU work.** Bench, GPU `--selftest`, GPU unit tests, microbenchmarks, and
profilers go through `.supervisor-action`. They are not started beside the
incumbent — a saturated GPU stalls that service even when the second job fits
in VRAM. `gpu-job` restores the incumbent. `engine-verify` replaces it only
on a pass. The CPU reference (`reference.py`) does not need a transaction.
