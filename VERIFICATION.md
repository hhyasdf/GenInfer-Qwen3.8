# Verification: qwen38-27b

Date: 2026-10-11. Hardware: 4070 Ti Super (sm_89, 16376 MiB) + 2080 Ti (sm_75, 22528 MiB,
Display GPU −1 G budget → 21504 MiB usable), PCIe, no NVLink; 29 GB RAM; offline.
Engine commit: `641b656` ("B5: ship the engine") — this file lands in that commit; the
verified tree is its working tree (clean at commit time; `gates --phase B5` re-run green
immediately before the commit). All evidence below is from 2026-10-11 runs in
`logs/supervisor.log`, `logs/debug-b5-smoke.log`, and the transactions they record.

## One-click

- clean-state run: **PASS**, time to serving: **96 s** (gpu-job 2026-10-11 01:17:22
  arg-parse → 01:18:58 `Uvicorn running on http://0.0.0.0:8081`), also **155 s** in the
  engine-verify transaction (00:46:43 start → 00:49:16 healthy).
- entry point: `.venv/bin/python -m minisgl` — zero arguments; every value (model dir,
  dtype, layer split 26/38, draft device, context 100k, backend triton_fa, cache naive)
  comes from `config.toml` and is printed in the startup log (`stack:` and
  `allocation table:`). The only CLI value in either run was the port, rendered by the
  supervisor to a spare port because the agent port was live.
- readiness probe: a real 1-token non-streaming `/v1/chat/completions` — 200 OK,
  2.660 s (engine-verify), and `/health` + `/v1/models` (both transactions).
- use-case round-trips (gpu-job 01:18:59–01:19:14, streamed + agent shapes):
  - streamed chat completion: 16 SSE chunks, first delta "We", last `finish_reason: stop`;
  - **agent-consumer turn**: assistant message with `content: null` and non-empty
    `tool_calls` (`get_weather`), followed by the `tool` result — 200, engine continued
    the turn, `finish_reason: stop`, coherent continuation
    ("The user is asking for the current weather in Paris. I have called the
    `get_weather` tool and received the result. I wil…");
  - non-streaming `/v1/completions`: 200, 7 tokens in 2.628 s.
- zero human intervention in either run; no edited files, no exported env vars, no
  installed tools. (The `MTMD_BACKEND_DEVICE`-style env prefix belongs to the origin
  command, not the engine.)

## Correctness

- determinism: **PASS** — 3 prompts × 30 greedy tokens, prefill logits max diff between
  runs 0.000e+00 on all 3 prompts (both ranks).
- golden test: bf16 engine vs baked `golden/logits.json` (independent HF reference
  oracle, layer-wise + greedy): logits max abs diff 2.280e-01 / 2.090e-01 / 1.948e-01
  ≤ 5.000e-01 (the file's tolerance; Q6_K weights against an unquantized reference),
  greedy mismatches **0/3** on every prompt (bf16 budget 3 per the tolerance table),
  page integrity PASS — no page leak. `--selftest` exit 0 in 108 s (GPU-exclusive,
  agent port verified free for the duration), run inside the engine-verify
  transaction 2026-10-11 00:44:55–00:46:43.
- golden model id equals the config model (gate check `check_golden_model`).
- smoke (http mode): the three round-trips above are the serving-mode smoke; outputs are
  coherent (quoted excerpts above). User sign-off: the user is directing this build and
  reads the engine's output; sign-off is this document.
- unit tests: `.venv/bin/python -m pytest tests/` — 10 passed.

## Performance

**Pending Tune** (gate 3 is the Tune track: measured at T0, closed at T3; B5 defers it
explicitly). What the fixed harness measured on 2026-10-11:

- engine, latency-1 (60000 prompt tokens / 4096 new, 5 runs + 2 warmup): the first
  request ran **600.113 s** on the server before the client aborted at its 300 s
  request timeout — the shared bench failed and the engine-verify transaction restored
  the origin. Short prompts decoded at **2.1–6.6 tok/s** (17 tokens / 8.052 s,
  31 tokens / 4.692 s, 7 tokens / 2.628 s).
- origin baseline (same harness, `origin-baseline` 2026-10-09, fields exactly match
  config `[bench]` — suite latency-1, 60000/4096, 5 runs, 2 warmup — so it is used as-is):
  TTFT p50 96653.868 ms / p99 96807.793 ms; decode tok/s p50 5070.762, p99 7.514. The
  p50 is an early-EOS artifact of the seeded random-token prompt (the model stops after
  a few hundred tokens on most runs); the p99 is the sustained-rate tail. Live origin
  traffic on 2026-10-11 decodes at **27.9 tok/s** with MTP spec-decode
  (draft acceptance 0.49, mean length 2.95).
- targets (SCENARIO.md): decode ≥ 20 tok/s @ concurrency 1; TTFT(60k) < 30 s.
  Both missed by a wide margin.
- bottleneck: **prefill-bound and decode-bound**. The engine runs no CUDA graphs
  ("CUDA graph is disabled" at startup), no spec-decode (the MTP block blk.64 is loaded
  but not yet consumed as a draft), and the triton_fa prefill path is the only attention
  backend that works on sm_75. T1 starts there.

## Agent port

- A live origin service (llama-server, pid 27359 — the post-baseline-restore instance of
  the user-confirmed command) owns 8080 and serves real traffic.
- `engine-verify` ran 2026-10-11 00:44:45–00:59:32: selftest PASS, engine serving on the
  spare port 8081, bench **lost** (600 s > 300 s client timeout) → the contract restored
  the origin. The engine binds the agent port only by winning `engine-verify`; that
  happens after Tune closes the performance gap (T3).
- Preflight guard (this ship): the engine now exits **2 before any model load** when its
  port is already being served (owner pid + remedy in the log). Verified live against
  busy 8080: exit 2 in 6.9 s, zero GPU footprint. The supervisor stray-engine scan
  remains the operational backstop for interactive launches.

## Gates

- `python -m minisgl.gates --phase B5` → `B5: ok` (check_not_toy, check_golden_model,
  check_origin_if_consumers).
