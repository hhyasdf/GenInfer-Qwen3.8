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

    def _prepare(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """The scheduler's ``_prepare_batch``, minus the ForwardInput wrapping."""
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        return input_mapping, write_mapping

    def run(
        self, text: str, tokenizer, sampling_params: SamplingParams
    ) -> Tuple[List[int], List[int]]:
        """One full request: prefill + greedy decode.

        Returns (prefill logits at the last position as a Python list,
        generated token ids).
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
        # model.forward() returns exactly [batch.size, vocab_size] rows: the
        # LM head slices prefill rows to the last position per request.
        batch = Batch(reqs=[req], phase="prefill")
        input_mapping, write_mapping = self._prepare(batch)
        batch.input_ids = self.token_pool[input_mapping]
        with torch.cuda.stream(engine.stream), engine.ctx.forward_batch(batch):
            logits = engine.model.forward()
        assert logits.shape[0] == batch.size, f"unexpected logits shape {tuple(logits.shape)}"
        prefill_logits = logits[0].float().cpu().tolist()
        next_token = int(logits[0].argmax().item())
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


def run_selftest(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m minisgl --selftest",
        description="Two-layer correctness gate: determinism + golden parity.",
    )
    parser.add_argument("--model-path", required=True, help="local model dir or HF model id")
    parser.add_argument(
        "--dtype", default="auto", help="auto (model's torch_dtype) | float16 | bfloat16 | float32"
    )
    parser.add_argument(
        "--golden",
        default=str(Path(__file__).resolve().parents[2] / "golden" / "logits.json"),
        help="path to the golden logits file (default: <engine>/golden/logits.json)",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=8,
        help="greedy tokens to generate and compare (must match the golden's greedy_ids length)",
    )
    parser.add_argument(
        "--cache-type", default="radix", choices=["radix", "naive"], help="KV cache type"
    )
    args = parser.parse_args(argv)

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

    dtype = _resolve_dtype(args.model_path, args.dtype)
    config = SchedulerConfig(
        model_path=args.model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=dtype,
        cache_type=args.cache_type,
        cuda_graph_max_bs=0,  # eager path — see module docstring
    )
    logger.info(
        "selftest: model=%s dtype=%s cache=%s decode_steps=%d prompts=%d",
        model_id,
        dtype,
        args.cache_type,
        args.decode_steps,
        len(prompts),
    )

    engine = Engine(config)
    runner = _PromptRunner(engine, config)
    tokenizer = load_tokenizer(args.model_path)

    # Greedy (the SamplingParams defaults), EOS ignored so the generated
    # length is exactly --decode-steps.
    sampling_params = SamplingParams(
        temperature=0.0, top_k=-1, top_p=1.0, ignore_eos=True, max_tokens=args.decode_steps
    )

    failures = 0
    for i, prompt in enumerate(prompts):
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
        if max_diff > tolerance:
            logger.error(
                "prompt %d (%r): LAYER 2 FAIL — prefill logits max abs diff %.3e > tolerance %.3e",
                i,
                text,
                max_diff,
                tolerance,
            )
            failures += 1
            continue
        g_tokens = list(prompt["greedy_ids"])
        if len(g_tokens) != args.decode_steps:
            logger.error(
                "prompt %d (%r): golden greedy_ids length %d != --decode-steps %d — "
                "regenerate the golden or pass the matching --decode-steps",
                i,
                text,
                len(g_tokens),
                args.decode_steps,
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
            tolerance,
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
    logger.info("SELFTEST PASS — %d prompts, determinism + golden parity, page integrity", len(prompts))
    return 0
