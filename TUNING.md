# TUNING.md — tuning ledger

One file per concern: this is the tuning ledger for `GenInfer-Qwen3.8`. It records
the T0 baseline, the T1 bottleneck diagnosis, and the before/after numbers for each
T2 change, until the B1 target is met or the user accepts the measured gap (T3).
The harness is the fixed ruler: the same `--bench` invocation scores the engine
before and after every change; no change to `client.py` / `perf.py` / `bench.py`.

## Target (B1, DESIGN.md:24)

- decode ≥ 20 tok/s sustained @ concurrency 1
- TTFT(60k) < 30 s

## T0 — baseline (no code change)

Fixed harness (`--base-url`, latency-1: 60000 prompt / 4096 new / 5 runs / 2 warmup),
run inside a supervisor `engine-verify` transaction against the engine as committed
(`2dd07c0`). Numbers below are the gate-suite record.

**Decode (step ring, `logs/step-ring.json`):**

| metric | value |
|---|---|
| decode steps recorded | 31 |
| step_ms min / median / max | 67.4 / 79.7 / 82.9 |
| graph_hit true | 0 / 31 |
| decode_batch | 1 |
| prefill_tokens (decode steps) | 0 |
| **sustained decode rate** | **12.5 tok/s** (1000 / 79.7) |

**60k request (engine-verify):** the request ran **600.113 s** and failed with the
OpenAI `AsyncOpenAI` SDK's default 600 s client timeout (`APITimeoutError`, message
"Request timed out."). It is **not** an engine hang: the request timestamps
(00:49:20 → 00:59:20) span exactly 600 s, and the prefill/decode decomposition shows
the engine was serving the whole time. Decomposition: prefill ≈ 272 s (≈ 220 tok/s)
+ 4096 decode steps at 12.5 tok/s ≈ 328 s = 600 s total. So the 60k request exceeds
the 600 s client budget because **total** request time (prefill + decode) exceeds it,
not because TTFT exceeds it. TTFT(60k) on the engine ≈ 272 s (the prefill portion).

**Short-prompt decode** (cold + warm, includes the small prefill): 2.1–6.6 tok/s.

**Origin reference** (`.origin-baseline.json`, the same fixed harness against the
user's llama-server, measured 2026-10-09): 60k TTFT p50 96.65 s, live decode
≈ 27.9 tok/s with MTP spec-decode (draft-mtp, n-max 4, acceptance ≈ 0.49).
`decode_tok_s_p50 = 5070.762` is the early-EOS artifact (a request that hits EOS
after a few tokens reports a huge tok/s; the distribution is bimodal, p99 7.514).
The honest origin decode rate is the live 27.9 tok/s.

## Gap vs B1

| target | engine | origin | gap |
|---|---|---|---|
| decode ≥ 20 tok/s | 12.5 tok/s | 27.9 tok/s | **1.6× short (−37.5%)** |
| TTFT(60k) < 30 s | ≈ 272 s | 96.6 s | **≈ 9× short** |

The decode gap is the closer, evidence-backed one. The TTFT gap is the harder,
likely-unreachable one (the engine prefill is ≈ 220 tok/s; 30 s would need ≈ 2000
tok/s, and the origin only reaches ≈ 620 tok/s). TTFT(60k) is expected to close at
T3 as a user-accepted gap unless the prefill regime changes.

## T1 — bottleneck (one regime named, with evidence)

**Regime: launch-bound decode.**

Evidence 1 — the step ring (the canonical regime evidence): `graph_hit: false` on
31/31 decode steps, `decode_batch: 1` (small batch, many small kernels — exactly the
launch-bound profile the tuning-card describes). The graph was never replayed.

Evidence 2 — the root cause, not a guess: `python/minisgl/engine/engine.py:179-180`
forces `graph_bs = []` and `graph_max_bs = None` whenever `config.uses_layer_split`,
despite `cuda_graph_max_bs = 1` being baked into `config.toml`. The reason is
structural: the layer-split rank handoff is a **blocking gloo send/recv**
(`_send_handoff` / `_recv_handoff`, `qwen35.py:818-835`) — a CPU collective that
sits **inside** `model.forward()` between the two layer halves. gloo send/recv are
host-side collectives, not capturable CUDA kernels, so a whole-forward capture
cannot span them. With `graph_bs = []`, `GraphRunner._determine_cuda_graph_bs`
returns `[]`, and every startup log confirms it: "CUDA graph is disabled." with
`graph_pool=0.00 GiB` (b4-verify, b5-oneclick, debug-b5-smoke, and every other run).

T2 hypothesis (one sentence): capture the two layer halves — front (embedding +
layers 0–25) and back (layers 26–63 + final norm + lm_head) — as CUDA graphs
**around** the gloo handoff, so the blocking CPU collective sits outside the
captured region and the ~50 ms/step of eager launch overhead collapses to one graph
replay per rank, pushing the decode step below the 50 ms/step (20 tok/s) line.

Second lever (recorded, not the T1 bottleneck): the MTP draft block `blk.64`
(`qwen35.py:761`, `MTPBlock`) is loaded on the back rank but **never consumed as a
draft** — that is the speculative-decode row, unlocked by DESIGN.md. If the graph
alone does not reach 20 tok/s, it is the next T1/T2 round.

Prefill regime (TTFT 272 s vs 30 s): a separate regime (prefill-bound), not the
bottleneck named here.

## T2 — change log

### Change 1 — layer-split CUDA graph capture (capture `forward_compute` around the gloo handoff)

**One change.** `engine.py:179-180` no longer forces `graph_bs = []` for layer-split.
`model.forward_compute(h=None, residual=None)` (`qwen35.py:818`) splits the forward
so the blocking gloo `_send_handoff` / `_recv_handoff` stays eager **outside** the
captured region: front rank captures `forward_compute()` (embedding + layers 0–25,
returns the `(h, residual)` tuple); back rank captures `forward_compute(h, residual)`
(layers 26–63 + norm + lm_head). The handoff tensors live in a `GraphCaptureBuffer`
(`graph.py`): the front graph writes `h`/`residual`, the back graph reads them (the
eager `_recv_handoff` fills them before `replay_back`).

Also fixed a real bug the capture exposed: the back rank's handoff receive was sized
by **request count** (`batch.size`) instead of **token count** (`input_ids.shape[0]`) —
`engine.py:407` now matches `qwen35.py:806`. The prefill batch (1 request, ~53 tokens)
crashed gloo (`542720 vs 10240` bytes) until this matched.

**Before (T0, eager, step ring 31 decode steps):**
- decode 12.5 tok/s, step_ms min/median/max 67.4 / 79.7 / 82.9, `graph_hit` 0/31.

**After (capture, step ring 7 decode steps — the bench request's decode portion):**
- decode 13.3 tok/s, step_ms min/median/max 58.5 / 75.4 / 77.8, `graph_hit` 7/7.
- capture log: "Start capturing CUDA graphs with sizes: [1, 2, 4]" → 3/3 done,
  `graph_pool=0.06 GiB`, "Free GPU memory after capturing CUDA graphs: 2.18 GiB".

**Verdict:** the graph replay works and lands on every decode step, but it only removed
~4.3 ms/step of launch overhead (79.7 → 75.4 ms). The remaining ~45 ms is the
bandwidth floor (13 GiB @ ~450 GB/s ≈ 30 ms) plus per-token gather/attention overhead.
20 tok/s needs ≤ 50 ms/step — the graph alone gets to ~75 ms. **Not enough.** The next
lever is MTP spec-decode (change 2), which is what crosses the line: the origin runs
`--spec-type draft-mtp --spec-draft-n-max 4` and the live origin log shows tg ≈
42–44 tok/s with draft acceptance 0.865.

**Bench note:** the 60k request ran 600.129 s and the `--bench` client timed out at the
`AsyncOpenAI` SDK's 600 s default (prefill ≈ 292 s + decode ≈ 308 s at 13.3 tok/s).
That is the harness 600 s client timeout, not an engine hang — and it is still bound
until decode clears 20 tok/s (then total < 600 s). See change 2.

### Change 2 — MTP draft spec-decode: duplication bug fixed, correctness now proven; perf still 13.1 tok/s

**What shipped.** The MTP spec path (1438+/105- across 14 files, layer-64 MTP head,
`mtp_n_max=4`, draft chain on the back rank) was already in place. This change is the
decisive correctness fix that was blocking it: the client was receiving duplicated,
interleaved text ("We need need to respond respond to…") while the token stream was
provably clean (EMIT = front-rank return = scheduler send = detokenizer input, each id
exactly once; offline decode of the full EMIT stream reads as coherent English).

**The bug (detokenizer).** `detokenize()` bundles **L+1 `DetokenizeMsg`s for one uid in
a single call** on the spec path, but the surrogate bookkeeping
(`surr_offset` / `read_offset`) was frozen during the id-collection loop, so message k's
`new_text = read_str[len(surr_str):]` decoded **all** batch tokens t1..tk — re-emitting
every earlier token of the same step. The standard sglang layout only works for the
1-message-per-uid-per-call normal path.

**The fix.** `python/minisgl/tokenizer/detokenize.py` now decodes each message against
its own boundary and advances the surrogate offsets between messages (per-message
`tokenizer.decode` instead of one `batch_decode` over stale offsets). Byte-identical to
the original for the normal 1-message path; correct for multi-message spec batches,
including the surrogate carry-over (a token whose decode ends in `\uFFFD` is held for
the next message exactly as before).

**Verification (`.spec-tokendump.sh`, tiny 120-token request).** Client output is now
fully coherent end-to-end: *"We need to respond to user: 'Write a short paragraph
describing a quiet river at dawn.' … A quiet river at dawn moves like a ribbon of pale
silver, its surface barely disturbed by the first breath of morning. Mist drifts low
over the water, softening the banks…"* The `[detok-out]` trace shows true incremental
deltas (`'drift'` then `'s'`, `'soft'` then `'ening'`) — no overlap.

**Performance (`hacks/spec-probe.sh`, clean — no debug env):**
- Request A (tiny prompt, 400 tokens): 30.949 s → **13.1 tok/s** sustained decode, coherent.
- Request B (~8k prompt, 50 tokens): 119.6 s wall → **TTFT(8k) ≈ 116 s** (prefill-dominated).
- Acceptance over 12 spec steps: 5×L=4, 4×L=0, 3×L=1 → **avg L ≈ 1.9**; 33% of steps
  emit only the bonus token.

**Verdict.** Decode **13.1 tok/s — no speedup** vs non-spec (13.3, change 1). Both targets
miss: decode needs ≥ 20, TTFT(60k) ≈ 272 s (T0) needs < 30 s. The gap is implementation
overhead, not draft quality: the origin `llama-server` runs the **same model + same MTP
head** at 42–44 tok/s with acceptance 0.865. Our spec step pays a 5-token verify prefill
via Triton FA over the growing KV cache, a TP=2 gloo handoff per step, **CUDA graph
disabled** on the spec path, and 2 forwards per step — while 33% of steps accept zero
draft tokens. The single-token decode floor is memory-bandwidth (~13 tok/s); only
multi-token-per-weight-read at high acceptance crosses it, which is exactly what the
origin's cheaper verify achieves. **Next lever:** cut per-step verify overhead — enable
CUDA graph on the spec path (the eager verify is the launch-overhead source) and/or drop
`mtp_n_max` so a low-acceptance step does not pay for a full 5-token verify.

**Bugs fixed this change:** the detokenizer duplication (above) and, in the same probe
cycle, the `ENGINE_RET_TRACE` reading uninitialized CPU memory (async `non_blocking=True`
copy not synced before `.tolist()` in `engine.py`) — a `torch.cuda.synchronize()` before
the trace makes the dump ids match EMIT exactly.
