"""Runtime self-test: the two-layer correctness gate (see references/verification.md).

Layer 1 — Determinism: the same prompt run twice yields an identical token
sequence. Catches unseeded RNGs and non-deterministic reduction order.

Layer 2 — Golden parity: the engine's prefill logits (last position) and its
greedy continuation are compared against ``golden/logits.json``
(format ``gen-inference-golden/v1``). The toy golden is float64. A real model
uses the three-level oracle in ``references/backends/cpu.md`` (one HF layer at
bf16 or fp32, then a short greedy). The golden file is baked into the engine;
the self-test verifies against it at runtime without the reference present.

The self-test drives the engine directly, with the exact construction path of
the server scheduler (``Engine`` + ``TableManager`` + ``CacheManager``) and
the scheduler's per-request recipe (prefill adder + ``_prepare_batch`` +
``_forward`` + ``_process_last_data``). CUDA graph capture is disabled
(``cuda_graph_max_bs=0``): prefill logits always come from the eager forward
(graphs apply to decode only), and decode goes through ``engine.forward_batch``
— the same code path the server uses. Graph replay is a mechanical capture of
the same forward and is exercised end-to-end by ``--bench`` (full server
stack). The second run of each prompt exercises the warm prefix cache (the
radix cache path), so both cache states are covered.

Exit codes: 0 = all pass, 1 = a check failed, 2 = usage/config error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch

from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.config import SchedulerConfig
from minisgl.scheduler.scheduler import _make_input_tuple, _make_positions, _make_write_tuple
from minisgl.scheduler.table import TableManager
from minisgl.scheduler.utils import PendingReq
from minisgl.utils import cached_load_hf_config, init_logger, load_tokenizer

logger = init_logger(__name__)

_GOLDEN_FORMAT = "gen-inference-golden/v1"


def canonical_model_id(model_path: str) -> str:
    """The engine's model id, as reported by ``GET /v1/models``.

    A local directory is canonicalized to its resolved absolute path (the
    supervisor renders config paths absolute before launch); a remote (HF)
    id is used as-is. ``reference.py`` must apply the same rule when it
    writes the golden file — no import, independence is the point.
    """
    p = Path(model_path)
    return str(p.resolve()) if p.is_dir() else model_path


def _resolve_dtype(model_path: str, dtype: str) -> torch.dtype:
    if dtype == "auto":
        return cached_load_hf_config(model_path).dtype
    resolved = getattr(torch, dtype, None)
    if resolved is None or not isinstance(resolved, torch.dtype):
        raise ValueError(f"unknown --dtype: {dtype!r} (expected auto|float16|bfloat16|float32)")
    return resolved


class _PromptRunner:
    """Drives one request through the engine with the scheduler's recipe.

    Mirrors, step for step:
      - ``scheduler/prefill.py`` ``_try_allocate_one`` / ``_add_one_req``
        (match + lock + table allocate + cached-part seeding + extend copy),
      - ``scheduler/scheduler.py`` ``_prepare_batch`` (pad, allocate pages,
        positions, input/write mappings, out_loc, attention metadata),
      - ``scheduler/scheduler.py`` ``_forward`` (token_pool read,
        ``engine.forward_batch``, token_pool write),
      - ``scheduler/scheduler.py`` ``_process_last_data`` (``append_host``)
        and ``_free_req_resources`` (table free, then cache insert).
    """

    def __init__(self, engine: Engine, config: SchedulerConfig):
        self.engine = engine
        self.device = engine.device
        self.table_manager = TableManager(config.max_running_req, engine.page_table)
        self.cache_manager = CacheManager(
            engine.num_pages, config.page_size, engine.page_table, config.cache_type
        )
        self.token_pool = self.table_manager.token_pool

    def _prepare(self, batch: Batch, full_prompt: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """The scheduler's ``_prepare_batch``, minus the ForwardInput wrapping.

        ``full_prompt=True`` (Fix B2): the GDN layers have no prefix cache —
        their delta-rule recurrence must replay the whole prompt from a zeroed
        state (the zeroing itself is Fix A, via Context.request_gdn_reset).
        So after allocating the suffix pages (which needs the real
        cached_len), temporarily zero cached_len so every forward mapping
        (positions, input/write tuples, out_loc, attention metadata) treats
        the batch as a cold full-prompt prefill (q_len == k_len, q_base = 0;
        the triton_fa kernel needs no change). Attention re-reads and
        re-writes the prefix KV through the page table with identical
        deterministic values; the GDN layers replay the full recurrence.
        cached_len is restored before returning.

        TODO(B1): replace this O(full_prompt) recompute with store/restore of
        the GDN recurrent state at the cache boundary — snapshot
        ctx.gdn_state[layer] once the cached prefix has been computed and
        restore it here, so the warm path replays only the suffix (mirroring
        what the attention KV prefix cache already does).
        """
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        saved_cached: list[int] | None = None
        if full_prompt:
            saved_cached = [r.cached_len for r in batch.reqs]
            for r in batch.reqs:
                r.cached_len = 0
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        if full_prompt:
            for r, c in zip(batch.reqs, saved_cached):
                r.cached_len = c
        return input_mapping, write_mapping

    def run(
        self, text: str, tokenizer, sampling_params: SamplingParams
    ) -> Tuple[List[int] | None, List[int]]:
        """One full request: prefill + greedy decode.

        Returns (prefill logits at the last position as a Python list,
        generated token ids). The logits are ``None`` on the front rank of a
        layer split: that rank runs only the lower layers, forwards the hidden
        states to the back rank, and receives the sampled next tokens back —
        the logits (and the golden parity built on them) live on the back rank,
        which is where ``model.forward()`` returns them.
        """
        engine = self.engine
        input_ids = tokenizer.encode(text, return_tensors="pt").view(-1).to(torch.int32)
        assert input_ids.is_cpu

        # ---- prefill adder (scheduler/prefill.py) ----
        pending = PendingReq(uid=0, input_ids=input_ids, sampling_params=sampling_params)
        handle = self.cache_manager.match_req(pending).cuda_handle
        cached_len = handle.cached_len
        self.cache_manager.lock(handle)
        table_idx = self.table_manager.allocate()
        if cached_len > 0:
            # Seed the cached prefix into the table row: tokens AND page
            # table (attention reads KV through the page table).
            self.token_pool[table_idx, :cached_len].copy_(
                input_ids[:cached_len].pin_memory(), non_blocking=True
            )
            self.table_manager.page_table[table_idx, :cached_len].copy_(
                handle.get_matched_indices()
            )
        extend = input_ids[cached_len:]
        self.token_pool[table_idx, cached_len : cached_len + extend.numel()].copy_(
            extend.pin_memory(), non_blocking=True
        )
        req = Req(
            input_ids=input_ids,
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=sampling_params.max_tokens,
            uid=0,
            sampling_params=sampling_params,
            cache_handle=handle,
        )

        # ---- prefill step: eager forward, capture last-position logits ----
        # model.forward() returns one logits row per REQUEST ([batch.size,
        # vocab_size]): the ParallelLMHead selects the last token's hidden
        # state per request (attn_metadata.get_last_indices) before the
        # lm_head projection, so each row is that request's last-position
        # logits. The cache manager always recomputes the final token
        # (match_req matches input_ids[: input_len - 1]), so this row is the
        # last-position logits for both cold and warm runs.
        # Fix A: this is a new request, so the global GDN recurrent state
        # (which holds the previous request's final recurrence) must be
        # zeroed before the prefill forward; Context.forward_batch consumes
        # the flag. Fix B2: on the warm path (cached_len > 0) the forward
        # replays the FULL prompt so the GDN recurrence covers the cached
        # prefix too — see _prepare(full_prompt=...).
        engine.ctx.request_gdn_reset()
        batch = Batch(reqs=[req], phase="prefill")
        input_mapping, write_mapping = self._prepare(batch, full_prompt=req.cached_len > 0)
        batch.input_ids = self.token_pool[input_mapping]
        with torch.cuda.stream(engine.stream), engine.ctx.forward_batch(batch):
            logits = engine.model.forward()
        if logits is None:
            # Front rank of a layer split: the forward sent the hidden states
            # to the back rank and returned no logits. The back rank samples
            # the next token and ships it back over the token chain.
            next_tokens_gpu = engine.model.recv_tokens(batch.size)
            next_token = int(next_tokens_gpu[0])
            prefill_logits = None
        else:
            assert logits.shape[0] == batch.size, f"unexpected logits shape {tuple(logits.shape)}"
            prefill_logits = logits[0].float().cpu().tolist()
            next_token = int(logits[0].argmax().item())
            if engine.config.uses_layer_split:
                # Back rank: ship the sampled token to the front rank
                # (forward_batch does the same exchange; prefill is driven
                # directly by model.forward(), so it does it here).
                engine.model.send_tokens(
                    torch.tensor([next_token], dtype=torch.int32, device=self.device)
                )
        req.complete_one()
        self.token_pool[write_mapping] = torch.tensor([next_token], dtype=torch.int32, device=self.device)
        req.append_host(torch.tensor([next_token], dtype=torch.int32))
        generated = [next_token]

        # ---- decode steps: engine.forward_batch (the server's decode path) ----
        for _ in range(sampling_params.max_tokens - 1):
            batch = Batch(reqs=[req], phase="decode")
            input_mapping, write_mapping = self._prepare(batch)
            batch.input_ids = self.token_pool[input_mapping]
            sample_args = engine.sampler.prepare(batch)
            with torch.cuda.stream(engine.stream):
                out = engine.forward_batch(batch, sample_args)
            out.copy_done_event.synchronize()
            self.token_pool[write_mapping] = out.next_tokens_gpu
            next_token = int(out.next_tokens_cpu[0])
            req.append_host(torch.tensor([next_token], dtype=torch.int32))
            generated.append(next_token)

        # ---- cleanup (scheduler's _free_req_resources order) ----
        self.table_manager.free(table_idx)
        self.cache_manager.cache_req(req, finished=True)

        return prefill_logits, generated


def _baked_config() -> dict:
    """Parse the engine's config.toml (``{}`` when absent)."""
    config_path = Path(__file__).resolve().parents[2] / "config.toml"
    if not config_path.is_file():
        return {}
    import tomllib

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _baked_attention_backend() -> str:
    """Read ``[performance].attention_backend`` from the engine's config.toml.

    The selftest builds its own SchedulerConfig (it bypasses the server's
    argument parsing), so it must pick up the baked backend itself. Without
    this the ``"auto"`` default selects a backend that can be incompatible
    with the target hardware (e.g. FlashInfer FA2 head_dim-256 prefill
    crashes on sm_75).
    """
    return _baked_config().get("performance", {}).get("attention_backend", "auto")


def _baked_dtype() -> str:
    """Read ``[engine].dtype`` from the engine's config.toml.

    The server's ``--dtype`` defaults to the baked value (``d("dtype",
    "auto")`` in ``server/args.py``); the selftest mirrors that. GGUF-only
    checkpoints have no config.json, so ``auto`` (the HF lookup) cannot
    resolve the precision.
    """
    return _baked_config().get("engine", {}).get("dtype", "auto")


def _baked_layer_split() -> Tuple[int, ...] | None:
    """Read ``[distributed].layer_split`` from the engine's config.toml.

    ``None`` (the default) is single-rank. A non-empty tuple splits the
    decoder across ranks (rank i runs ``sum(split[:i]) .. sum(split[:i+1])``),
    each rank bound to ``cuda:{rank}``, cross-rank traffic over gloo. The
    selftest must mirror the server's split or the model does not fit on a
    single GPU (21.6 GiB on a 16 GiB card).
    """
    ls = _baked_config().get("distributed", {}).get("layer_split")
    if not ls:
        return None
    return tuple(int(x) for x in ls)


def run_selftest(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m minisgl --selftest",
        description="Two-layer correctness gate: determinism + golden parity.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="local model dir or HF model id (default: the model baked in <engine>/config.toml "
        "[engine].model — the selftest verifies the same model the server serves)",
    )
    parser.add_argument(
        "--dtype",
        default=_baked_dtype(),
        help="auto (model's torch_dtype) | float16 | bfloat16 | float32 "
        "(default: config.toml [engine].dtype, or 'auto')",
    )
    parser.add_argument(
        "--golden",
        default=str(Path(__file__).resolve().parents[2] / "golden" / "logits.json"),
        help="path to the golden logits file (default: <engine>/golden/logits.json)",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=None,
        help="greedy tokens to generate and compare (default: the golden's greedy_ids length; "
        "a mismatch is a hard failure — a selftest that skips its golden layer is a trap)",
    )
    parser.add_argument(
        "--cache-type", default="radix", choices=["radix", "naive"], help="KV cache type"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="CUDA device index to load the model on (default: derived from the rank, cuda:0). "
        "Set this when the model does not fit on cuda:0.",
    )
    parser.add_argument(
        "--num-pages",
        type=int,
        default=None,
        help="override the KV-cache page count (default: auto-sized from free memory). "
        "Set a small value when the model fills most of the GPU so the auto-sizing "
        "does not over-reserve.",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="attention backend (default: config.toml [performance].attention_backend, "
        "or 'auto' when config.toml has no such field)",
    )
    args = parser.parse_args(argv)

    # ---- model path: CLI flag, else the model baked into config.toml ----
    # The zero-argument invariant (`python -m minisgl --selftest` passes) means
    # the selftest must verify the model the server serves without the caller
    # remembering its path — the same config.toml [engine].model the cmd entry
    # point and the supervisor use. Defaulting to a toy path instead would
    # silently verify the wrong model in an engine whose real checkpoint lives
    # elsewhere (and fail the stale-golden check on top).
    if args.model_path is None:
        import tomllib

        config_path = Path(__file__).resolve().parents[2] / "config.toml"
        if not config_path.is_file():
            logger.error("no --model-path and no %s — pass --model-path explicitly", config_path)
            return 2
        with open(config_path, "rb") as f:
            model = tomllib.load(f)["engine"]["model"]
        mp = Path(model)
        # A relative model dir is resolved against the engine root (the
        # supervisor renders config paths absolute before launch for the same
        # reason — the stale-golden check needs the canonical absolute id,
        # which is_dir() can only produce from a real path).
        if not mp.is_dir() and (config_path.parent / mp).is_dir():
            mp = (config_path.parent / mp).resolve()
        args.model_path = str(mp)
        logger.info("model path from config.toml: %s", args.model_path)

    # ---- golden file (Layer 2 input; also defines the prompt set) ----
    golden_path = Path(args.golden)
    if not golden_path.is_file():
        logger.error("golden file not found: %s", golden_path)
        return 2
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)
    if golden.get("format") != _GOLDEN_FORMAT:
        logger.error(
            "unsupported golden format: %r (expected %s)", golden.get("format"), _GOLDEN_FORMAT
        )
        return 2
    model_id = canonical_model_id(args.model_path)
    if golden.get("model") != model_id:
        logger.error(
            "stale golden file: golden model %r != engine model id %r — regenerate golden/logits.json",
            golden.get("model"),
            model_id,
        )
        return 2
    tolerance = float(golden["tolerance"]["max_abs_diff"])
    prompts = golden.get("prompts") or []
    if not prompts:
        logger.error("golden file has no prompts")
        return 2

    # ---- decode steps: default to the golden's length, never skip the golden layer ----
    # The trap this defeats: a selftest invoked without --decode-steps against a
    # golden whose greedy_ids differ from the CLI default either compares the
    # wrong number of tokens or (in older engines) silently skips layer 2 while
    # printing SELFTEST PASS. Deriving the default from the golden file makes a
    # bare --selftest always exercise the full golden layer; an explicit
    # --decode-steps that mismatches is a hard failure below, not a skip.
    golden_len = len(prompts[0]["greedy_ids"])
    if any(len(p["greedy_ids"]) != golden_len for p in prompts):
        logger.error("golden file prompts have inconsistent greedy_ids lengths — regenerate golden/logits.json")
        return 2
    if args.decode_steps is None:
        args.decode_steps = golden_len
        logger.info("decode_steps unset — derived %d from the golden file", golden_len)

    layer_split = _baked_layer_split()
    world_size = len(layer_split) if layer_split is not None else 1
    if world_size > 1 and args.gpu is not None:
        logger.warning(
            "selftest: --gpu ignored under a layer split (each rank binds cuda:{rank})"
        )
    state = {
        "model_path": args.model_path,
        "dtype_str": args.dtype,
        "cache_type": args.cache_type,
        "decode_steps": args.decode_steps,
        "prompts": prompts,
        "tolerance": tolerance,
        "num_pages": args.num_pages,
        "attention_backend": args.attention_backend or _baked_attention_backend(),
        "gpu": args.gpu,
        "layer_split": layer_split,
    }
    if world_size > 1:
        return _spawn_selftest(world_size, state)
    return _run_selftest_rank(0, 1, state)


def _spawn_selftest(world_size: int, state: dict) -> int:
    """Spawn one selftest process per rank and aggregate the exit codes.

    The parent does no GPU work (mirroring ``server/launch.py``): each rank
    builds its own ``Engine`` with ``DistributedInfo(rank, world_size)`` and
    the shared gloo rendezvous, runs the two-layer gate on its slice, and
    exits 0/1/2. The back rank's layer-2 golden parity transitively covers
    the front rank's layers (its logits consume the front rank's handoff
    activations).
    """
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    procs = [
        mp.Process(
            target=_run_selftest_rank,
            args=(i, world_size, state),
            name=f"selftest-rank{i}",
        )
        for i in range(world_size)
    ]
    for p in procs:
        p.start()
    rc = 0
    for p in procs:
        p.join()
        if p.exitcode not in (0, None):
            logger.error("selftest: rank process exited with code %d", p.exitcode)
        rc = max(rc, p.exitcode or 0)
    return rc


def _run_selftest_rank(rank: int, world_size: int, state: dict) -> int:
    """Run the two-layer gate on one rank of the (possibly split) engine.

    Single-rank (``world_size == 1``): the historical in-process path —
    ``cuda:{rank}`` (rank 0) or an explicit ``--gpu``. Split
    (``world_size > 1``): this process is rank ``rank`` of the layer split;
    the device binds to ``cuda:{rank}``, cross-rank traffic runs over the
    gloo group, and the per-rank slice comes from the shared ``layer_split``.
    """
    dtype = _resolve_dtype(state["model_path"], state["dtype_str"])
    attention_backend = state["attention_backend"]
    device = None if world_size > 1 else (f"cuda:{state['gpu']}" if state["gpu"] is not None else None)
    config = SchedulerConfig(
        model_path=state["model_path"],
        tp_info=DistributedInfo(rank, world_size),
        dtype=dtype,
        cache_type=state["cache_type"],
        cuda_graph_max_bs=0,  # eager path — see module docstring
        device=device,
        num_page_override=state["num_pages"],
        attention_backend=attention_backend,
        layer_split=state["layer_split"],
    )
    device_str = (
        f"cuda:{rank}"
        if world_size > 1 or state["gpu"] is None
        else f"cuda:{state['gpu']}"
    )
    logger.info(
        "selftest rank %d/%d: model=%s dtype=%s cache=%s attn=%s decode_steps=%d "
        "prompts=%d device=%s num_pages=%s",
        rank,
        world_size,
        canonical_model_id(state["model_path"]),
        dtype,
        state["cache_type"],
        attention_backend,
        state["decode_steps"],
        len(state["prompts"]),
        device_str,
        state["num_pages"] if state["num_pages"] is not None else "auto",
    )

    engine = Engine(config)
    runner = _PromptRunner(engine, config)
    tokenizer = load_tokenizer(state["model_path"])

    # Greedy (the SamplingParams defaults), EOS ignored so the generated
    # length is exactly --decode-steps.
    sampling_params = SamplingParams(
        temperature=0.0, top_k=-1, top_p=1.0, ignore_eos=True, max_tokens=state["decode_steps"]
    )

    failures = 0
    for i, prompt in enumerate(state["prompts"]):
        text = prompt["text"]

        # Tokenizer drift check: the golden records the reference's encoding
        # of the same text; a different tokenizer.json would shift every id.
        engine_ids = list(tokenizer.encode(text))
        if engine_ids != list(prompt["ids"]):
            logger.error(
                "prompt %d (%r): tokenizer drift — engine ids %s… != golden ids %s…",
                i,
                text,
                engine_ids[:16],
                list(prompt["ids"])[:16],
            )
            failures += 1
            continue

        # ---- Layer 1: determinism (run 2 exercises the warm prefix cache) ----
        logits1, tokens1 = runner.run(text, tokenizer, sampling_params)
        logits2, tokens2 = runner.run(text, tokenizer, sampling_params)
        if tokens1 != tokens2:
            logger.error(
                "prompt %d (%r): LAYER 1 FAIL — non-deterministic tokens:\n  run1 %s\n  run2 %s",
                i,
                text,
                tokens1,
                tokens2,
            )
            failures += 1
        else:
            if logits1 is None:
                # Front rank: the tokens are deterministic because the back
                # rank's tokens are; the logits themselves live on the back
                # rank, so there is nothing to diff here.
                logger.info(
                    "prompt %d (%r): layer 1 PASS — deterministic (%d tokens; "
                    "front rank — prefill logits live on the back rank)",
                    i,
                    text,
                    len(tokens1),
                )
            else:
                logits_diff = max(abs(a - b) for a, b in zip(logits1, logits2))
                logger.info(
                    "prompt %d (%r): layer 1 PASS — deterministic (%d tokens; "
                    "prefill logits max diff between runs %.3e)",
                    i,
                    text,
                    len(tokens1),
                    logits_diff,
                )

        # ---- Layer 2: golden parity ----
        if logits1 is None:
            # Front rank: golden parity is asserted on the back rank, which
            # consumes this rank's handoff activations — a corrupted lower
            # layer would change the back rank's logits and fail its L2.
            logger.info(
                "prompt %d (%r): layer 2 SKIP — front rank (golden parity runs "
                "on the back rank, which consumes this rank's handoff)",
                i,
                text,
            )
            continue
        g_logits = torch.tensor(prompt["logits"], dtype=torch.float32)
        e_logits = torch.tensor(logits1, dtype=torch.float32)
        if e_logits.shape != g_logits.shape:
            logger.error(
                "prompt %d: golden logits shape %s != engine logits shape %s",
                i,
                tuple(g_logits.shape),
                tuple(e_logits.shape),
            )
            failures += 1
            continue
        max_diff = float((e_logits - g_logits).abs().max().item())
        if max_diff > state["tolerance"]:
            logger.error(
                "prompt %d (%r): LAYER 2 FAIL — prefill logits max abs diff %.3e > tolerance %.3e",
                i,
                text,
                max_diff,
                state["tolerance"],
            )
            failures += 1
            continue
        g_tokens = list(prompt["greedy_ids"])
        if len(g_tokens) != state["decode_steps"]:
            logger.error(
                "prompt %d (%r): golden greedy_ids length %d != --decode-steps %d — "
                "regenerate the golden or pass the matching --decode-steps",
                i,
                text,
                len(g_tokens),
                state["decode_steps"],
            )
            failures += 1
            continue
        budget = int(prompt.get("greedy_max_mismatches", 0))
        mismatches = sum(1 for a, b in zip(tokens1, g_tokens) if a != b)
        if mismatches > budget:
            logger.error(
                "prompt %d (%r): LAYER 2 FAIL — greedy mismatches %d > budget %d",
                i,
                text,
                mismatches,
                budget,
            )
            failures += 1
            continue
        logger.info(
            "prompt %d (%r): layer 2 PASS — logits max abs diff %.3e <= %.3e; "
            "greedy mismatches %d/%d",
            i,
            text,
            max_diff,
            state["tolerance"],
            mismatches,
            budget,
        )

    # ---- page leak check: after all cleanup, free + cached pages sum to num_pages ----
    try:
        runner.cache_manager.check_integrity()
        logger.info("page integrity PASS — no page leak")
    except RuntimeError as e:
        logger.error("page integrity FAIL — %s", e)
        failures += 1

    if failures:
        logger.error("SELFTEST FAIL — %d failure(s)", failures)
        return 1
    logger.info("SELFTEST PASS — %d prompts, determinism + golden parity, page integrity", len(state["prompts"]))
    return 0
