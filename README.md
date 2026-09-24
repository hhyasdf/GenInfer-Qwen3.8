# engine-scaffold

The starting point for every engine this skill generates: a complete,
self-testable Python inference engine that runs a **toy model** (byte
tokenizer, 2-layer Llama decoder) so the whole stack — build, serve, verify,
tune — works before any real model is integrated. The skill copies this
directory and points it at the real model.

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
(default 8080). This is the user-facing one-click entry point — the cmd entry
point, run from the venv, with zero arguments.

**One execution example.** A use-case request the engine must serve:

```bash
curl -sN localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"models/tiny-llama","messages":[{"role":"user","content":"hello"}],"max_tokens":8}'
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
before re-running; the previous post-mortem is `.prev`. A run on the toy
config does not prove a real model.

```bash
curl -s localhost:8080/health
curl -s localhost:8080/v1/models
curl -sN localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"models/tiny-llama","messages":[{"role":"user","content":"hello"}],"max_tokens":8}'
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
`references/verification.md` in the skill repo):

1. **Determinism** — same prompt + same seed ⇒ identical tokens. Always runs.
2. **Golden parity** — the forward pass is compared against
   `golden/logits.json`, produced by `reference.py`, an *independent* Python
   float64 implementation of the toy model (HF transformers on CPU). Logits
   must match within the file's recorded tolerance and the greedy continuation
   within its recorded mismatch budget. Skipped (not failed) when the golden
   file is absent.

```bash
.venv/bin/python -m minisgl --selftest
# determinism PASS (8 tokens, seeded)
# golden PASS (4 prompts, max_abs_diff=... <= 1.000e-03, greedy mismatches=0)
# selftest PASS (tiny-engine)
```

Regenerate the golden file after changing the toy model or its weights:

```bash
python3 reference.py
```

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
| `python/minisgl/` | the engine package: the mini-sglang code tree + the contract layer (`admin.py`, `completions.py`, `selftest.py`, `bench.py`, `__main__.py`) |
| `reference.py` | toy float64 reference; writes `golden/logits.json` |
| `golden/` | the trusted reference artifact for `--selftest` |
| `models/tiny-llama/` | the scaffold's toy model (the skill replaces it with the real checkpoint) |
| `pyproject.toml` | the package definition (editable install; `package-dir = {"" = "python"}`) |

## Memory (pre-allocate, don't grow)

Every engine this skill generates pre-allocates all of its memory — weights,
KV cache, activation workspace — once at startup, sized from the baked-in
max_context × max_running_requests. The serving loop allocates nothing: a
sequence is a cursor into a pre-allocated pool, never a growing buffer. The
toy model (a real 2-layer Llama decoder) has a KV cache, so the contract is
visible end-to-end: the engine's `CacheManager` pre-allocates the KV pool at
startup, and the scheduler's serving loop allocates nothing. When the real
model lands (B3–B4) the memory plan is pre-allocated per that rule, and
the engine prints the full allocation table at startup and fails fast with the
arithmetic when the pools do not fit.

## What the skill replaces

- `models/tiny-llama/` (the toy model) → the real model's checkpoint (the
  skill fills `config.toml`'s `model` field with the real checkpoint path).
- `reference.py` + `golden/` → the real model's reference (e.g. HF
  transformers) and its golden logits.
- `config.toml` → every field filled from the scenario profile.
- The CUDA backend (mini-sglang) stays — it is the target backend (the skill
  does not port a new backend; it uses mini-sglang's CUDA backend as-is).
