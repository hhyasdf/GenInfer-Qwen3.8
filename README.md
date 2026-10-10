# qwen38-27b engine

The 0-to-1 inference engine the gen-inference skill built for this scenario:
**Qwen3.8-27B** (GDN hybrid, GGUF Q6_K 21.58 GiB including the embedded MTP
block blk.64, mmproj Q8_0 for vision) on **4070 Ti Super + 2080 Ti** (PCIe,
no NVLink), serving an **OpenAI-compatible API** at concurrency 1 for a single
user + a tuning agent. Everything is baked into `config.toml`; the evidence
for every gate below is in `VERIFICATION.md`. Scenario: `SCENARIO.md`;
design: `DESIGN.md`; tuning record (pending): `TUNING.md`.

The engine is `minisgl` — a Python/torch inference engine copied from
[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang) (MIT),
installed editable into `.venv`. It is a FastAPI/uvicorn frontend plus one
scheduler worker per GPU plus tokenizer/detokenizer workers, all children of
the frontend process. The target backend is CUDA (mini-sglang is CUDA-only);
the correctness oracle is independent of the engine (float64 for the toy;
a real model uses one HF layer at bf16 or fp32, then a short greedy)
(`reference.py`), not a backend inside the engine.

## One command

```bash
.venv/bin/python -m minisgl
```

Serves the OpenAI-compatible API on the port baked into `config.toml`
(8080 — the agent port; the engine takes it by winning `engine-verify`, see
`VERIFICATION.md` § Agent port). This is the user-facing one-click entry point
— the cmd entry point, run from the venv, with zero arguments. Startup prints
the `stack:` line and the full `allocation table:` (weights 12.90 GiB, KV
3.48 GiB, page table, activation workspace) and fails fast with the
arithmetic if the pools do not fit the Display-GPU budgets.

**One execution example.** A use-case request the engine must serve (streamed
chat completion; the model field is the engine's model id):

```bash
curl -sN localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"hello"}],"max_tokens":8,"stream":true}'
```

**Supervised entry (skill debugging).** The skill's supervisor
(`templates/supervisor/run.sh <engine-dir>`, relative to the skill's repo) is the
outage transaction used during skill execution and debugging — it `cd`s to
this engine directory and `exec`s the skill's `supervisor.py` (with this engine
directory as its working directory). It waits until `.supervisor-action`
names a `gpu-job`, `origin-baseline`, or `engine-verify` (see below).

**Where the supervisor lives.** The supervisor entry (`run.sh`) and the
supervisor itself (`supervisor.py`, stdlib only) both live in the gen-inference
skill (`templates/supervisor/`, installed into the agent via a symlink — e.g.
`~/.pi/agent/skills/gen-inference`); `run.sh` resolves `supervisor.py` by
relative position (`$0`), so the binding works from any install location. The
engine carries no supervisor files of its own. `run.sh <engine-dir>` is a thin wrapper: it `cd`s to the engine
directory and `exec`s the skill's `supervisor.py` from there. The supervisor
logic lives in the skill, **not** in this directory: a generated engine carries
no supervisor files of its own (one fix in the skill updates every engine it
supervises). The supervisor reads `config.toml` from the engine directory.

**Quality flags.** The quality parameters (model path, sampling) are baked
into `config.toml` as defaults and overridable at runtime without a rebuild:

```bash
.venv/bin/python -m minisgl --temperature 0.8 --top-p 0.9 --model /path/to/checkpoint
```

Precedence: per-request OpenAI fields > CLI flags > `config.toml`. The
performance coefficients (context, batching, backend, port) are not settable —
they are baked in and printed in the startup log. Flags passed to
`run.sh <engine-dir>` (other than `--rebuild`) are forwarded when the
supervisor starts the engine.

On startup the previous `logs/supervisor.log` is renamed to
`logs/supervisor.log.prev` and a fresh log is opened. Read the current file
before re-running; the previous post-mortem is `.prev`.

```bash
curl -s localhost:8080/health
curl -s localhost:8080/v1/models
curl -sN localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"hello"}],"max_tokens":8,"stream":true}'
```

**Request contract.** `messages[].content` accepts the full OpenAI shape: a
string, `null` (an assistant turn that carries `tool_calls`), a missing field,
or a content-parts array. `null`/missing become an empty string; a parts array
serves its text parts joined (non-text parts — images, unknown types — are
dropped with a warning log, since the engine serves text). Every request is
logged on receipt and on completion (tokens, elapsed, tok/s); a request that
fails to parse is rejected with `422` **and** logged with a 256-char body
preview — no more silent 422.

## Supervisor

The skill's `templates/supervisor/run.sh <engine-dir>` waits. It does not
kill the agent port on startup and does not start this engine until a
transaction says to. Actions are a JSON file, `.supervisor-action`:

- `gpu-job` — run a command that does not bind the agent port, then restore whoever was serving.
- `origin-baseline` — start `[origin].command` on the spare port, measure it, restore the incumbent.
- `engine-verify` — measure this engine on the spare port. It binds the agent port only when it serves and beats `.origin-baseline.json`.

A GPU job that would stall the incumbent (bench, selftest, a second server, a profiler) is one of these actions. Readiness is a 1-token chat completion. The protocol is `references/tuning.md` in the skill repo.

```toml
[origin]
command = "llama-server --model models/model.gguf --port {port}"
```

## Verification

`--selftest` runs two layers (protocol and tolerances:
`references/verification.md` in the skill repo). The selftest is a single
layer-split process that loads the full model across both GPUs; it needs the
GPUs exclusive and the agent port free, so it runs inside a supervisor
transaction:

1. **Determinism** — same prompt + same seed ⇒ identical tokens. Always runs.
   3 prompts × 30 greedy tokens, prefill logits max diff between runs
   0.000e+00.
2. **Golden parity** — the forward pass is compared against
   `golden/logits.json`, baked from an *independent* HF reference (one layer
   at bf16, then a short greedy), not a backend inside the engine. Logits
   must match within the file's recorded tolerance (max abs diff 5e-1 for a
   Q6_K engine against the unquantized reference) and the greedy continuation
   within its recorded mismatch budget. Result: logits max abs diff
   0.195–0.228, greedy mismatches 0/3 on every prompt, page integrity clean.
   The full record, with the exact runs: `VERIFICATION.md`.

The golden file records `model: <the absolute model path>`; `--selftest`
fails on a model-id mismatch, so a golden file baked for a different model is
an error, not a pass.

## Benchmark

`--bench` is the fixed benchmark harness of the tuning loop — the same
invocation before and after every change. With no incumbent it starts a
server, waits until a 1-token chat completion succeeds, measures, and tears
the server down:

```bash
.venv/bin/python -m minisgl --bench
```

`--base-url` measures a server that is already up and does not bind a port.
The supervisor uses that form, against the spare port, for both the origin
service and the engine. `--suite` is `latency-1`, `mixed`, or `prefix-heavy`;
`[bench].suite` is the one the replacement gate compares. The summary line is
`BENCH-SUMMARY` plus a JSON object (`--summary-out` writes the same object to
a file), including p50 and p99.

## Layout

| Path | Role |
|---|---|
| `AGENTS.md` | the engine's "read this first": the scenario's key information (filled by the skill in B3) |
| `.incumbent` | `origin` or `engine` — who the supervisor restores |
| `.engine.pid` | supervisor-written pid of the active child (stale = the supervisor died) |
| `.engine-code.prev/` | supervisor-written snapshot of the last known-good code tree (rollback target) |
| `logs/supervisor.log` | the supervisor's persistent log (its own lines + the child's output); the post-mortem for a dead session |
| `config.toml` | the baked-in config: performance coefficients (context, batching, backend, port, health window) + quality defaults (sampling, model path — CLI-overridable at runtime) |
| `python/minisgl/` | the engine package: the mini-sglang code tree + the contract layer (`admin.py`, `completions.py`, `selftest.py`, `bench.py`, `__main__.py`) + the qwen35 model family (GDN hybrid decoder, MTP draft block, CLIP mmproj) |
| `reference.py` | the scaffold's toy float64 reference (writes the toy `golden/logits.json` for `pytest`) |
| `golden/logits.json` | the trusted reference artifact for `--selftest`, baked from the independent HF reference (`hacks/bake_golden`): 3 prompts × greedy 30 + 248320-vocab logits, bf16 tolerance |
| `hacks/` | ad-hoc test/debug scripts — the per-family weight-load checks, the GDN kernel oracles, and `bake_golden` (the HF reference that produced the golden file); not part of the engine |
| `pyproject.toml` | the package definition (editable install; `package-dir = {"" = "python"}`) |
| `VERIFICATION.md` | the B5 ship record: one-click, golden parity, bench gap, agent-port disposition |

## Memory (pre-allocate, don't grow)

Every engine this skill generates pre-allocates all of its memory — weights,
KV cache, activation workspace — once at startup, sized from the baked-in
max_context × max_running_requests. The serving loop allocates nothing: a
sequence is a cursor into a pre-allocated pool, never a growing buffer. This
engine prints the full allocation table at startup (weights 12.90 GiB, KV
3.48 GiB, activation workspace, page table) and fails fast with the
arithmetic when the pools do not fit the Display-GPU budgets (2080 Ti budget
21504 MiB = total − 1 G display reserve).

## What the skill replaced (from the scaffold)

- `models/tiny-llama/` (the toy model) → the real checkpoint
  `~/Models/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2/`
  (Q6_K main + Q8_0 mmproj), with `config.toml`'s `model` field filled from it.
- The toy float64 reference → the independent HF reference (`hacks/bake_golden`)
  and its baked `golden/logits.json`.
- `config.toml` → every field filled from the scenario profile.
- The CUDA backend (mini-sglang) stays — it is the target backend (the skill
  does not port a new backend; it uses mini-sglang's CUDA backend as-is).
