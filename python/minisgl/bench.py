"""Fixed benchmark harness: the ``--bench`` invocation of the tuning loop.

The same invocation before and after every change; the numbers go into
TUNING.md (see references/tuning.md). It exercises the *real serving path* —
the full multi-process stack (uvicorn frontend + scheduler workers +
tokenizer/detokenizer, exactly as ``launch_server`` builds it) — and drives
it over HTTP with the OpenAI client, so API latency, scheduling, batching,
and CUDA graph replay are all measured, not just the bare model forward.

Workload (fixed by the engine's config.toml ``[bench]`` section, rendered
into these flags by the skill's supervisor): N requests of ``prompt_tokens``
input tokens (built by this engine's tokenizer) and ``new_tokens`` output
tokens each (temperature 0, plain OpenAI body), ``warmup`` warmup requests
dropped, ``runs`` measured. The prompt sequence is seeded, so two runs with
the same tokenizer and the same flags send the same text. That is what makes
an origin server and this engine comparable.

Usage:
  ``python -m minisgl --bench --model-path <path> [flags]``
      start a server, measure it, tear it down. Only when no incumbent is
      using the GPU.
  ``python -m minisgl --bench --base-url http://host:port --model-path <path>``
      measure a server that is already up. This is the ruler the supervisor
      uses for both the origin service and the engine, so the numbers compare.
Exit codes: 0 = completed, 1 = server not serving or benchmark failed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

import torch
import uvicorn

from minisgl.benchmark.client import (
    benchmark_one,
    generate_prompt,
    process_benchmark_results,
)
from minisgl.distributed import DistributedInfo
from minisgl.server.args import ServerArgs
from minisgl.utils import cached_load_hf_config, init_logger, load_tokenizer

logger = init_logger(__name__, "bench")


def _resolve_dtype(model_path: str, dtype: str) -> torch.dtype:
    if dtype == "auto":
        return cached_load_hf_config(model_path).dtype
    resolved = getattr(torch, dtype, None)
    if resolved is None or not isinstance(resolved, torch.dtype):
        raise ValueError(f"unknown --dtype: {dtype!r} (expected auto|float16|bfloat16|float32)")
    return resolved


def _model_id(base: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:
            data = json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None
    rows = data.get("data") if isinstance(data, dict) else None
    if rows and isinstance(rows[0], dict) and rows[0].get("id"):
        return str(rows[0]["id"])
    return None


def _wait_serving(base: str, timeout_s: float) -> str | None:
    """Return the server's model id once a 1-token chat completion succeeds.

    ``GET /health`` is not proof the model serves.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        model = _model_id(base)
        if model:
            body = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "temperature": 0,
            }).encode()
            req = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=body,
                headers={"content-type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=max(1.0, deadline - time.time())) as r:
                    if 200 <= r.status < 300:
                        return model
            except (urllib.error.URLError, OSError, ValueError):
                pass
        time.sleep(0.5)
    return None


def _kill_worker_processes() -> None:
    """Terminate the spawned backend workers (same pattern as the shell REPL)."""
    import psutil

    parent = psutil.Process()
    for child in parent.children(recursive=True):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(parent.children(recursive=True), timeout=10)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def _at(xs: List[float], q: float) -> float:
    if not xs:
        raise ValueError("empty")
    idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * q))))
    return xs[idx]


def _emit_summary(summary: dict, path: str | None) -> None:
    print("BENCH-SUMMARY " + json.dumps(summary), flush=True)
    if path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")


def _measure(base: str, model: str, args) -> dict | None:
    """Same request shape against any OpenAI server already listening at ``base``."""
    from openai import AsyncOpenAI

    tokenizer = load_tokenizer(args.model_path)
    # Same seed + same tokenizer + same suite ⇒ the same prompt text on every invocation.
    random.seed(0)
    shared_prefix = generate_prompt(tokenizer, args.prompt_tokens) if args.suite == "prefix-heavy" else ""
    mixed_short = generate_prompt(tokenizer, max(8, args.prompt_tokens // 4)) if args.suite == "mixed" else ""
    mixed_long = generate_prompt(tokenizer, args.prompt_tokens) if args.suite == "mixed" else ""

    async def _bench_loop() -> List:
        raw: List = []
        client = AsyncOpenAI(base_url=f"{base}/v1", api_key="minisgl")
        for i in range(args.warmup + args.runs):
            if args.suite == "prefix-heavy":
                prompt = shared_prefix
            elif args.suite == "mixed":
                prompt = mixed_short if i % 2 == 0 else mixed_long
            else:
                prompt = generate_prompt(tokenizer, args.prompt_tokens)
            rr = await benchmark_one(
                client,
                prompt,
                args.new_tokens,
                model,
                pbar=False,
                vendor_ext=False,
            )
            if i >= args.warmup:
                raw.append(rr)
                logger.info("measured %d/%d", len(raw), args.runs)
        return raw

    try:
        raw_results = asyncio.run(_bench_loop())
    except Exception as e:
        logger.error("benchmark failed: %s", e)
        return None
    if not raw_results:
        logger.error("no measured results — benchmark FAILED")
        return None
    try:
        process_benchmark_results(raw_results, tokenizer=tokenizer)
    except Exception as e:
        logger.error("benchmark stats failed: %s", e)
        return None

    tics = [r.tics for r in raw_results]
    if any(len(t) < 2 for t in tics):
        logger.error("a measured request produced no token")
        return None
    ttft = sorted(t[1] - t[0] for t in tics)
    tpot = sorted(d for t in tics for d in (t[i + 1] - t[i] for i in range(1, len(t) - 1)))
    e2e = sorted(t[-1] - t[0] for t in tics)
    if not tpot:
        logger.error("decode gaps are empty; raise --new-tokens above 1")
        return None
    summary = {
        "suite": args.suite,
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "runs": len(raw_results),
        "warmup": args.warmup,
        "ttft_p50_ms": round(_at(ttft, 0.50) * 1000, 3),
        "ttft_p99_ms": round(_at(ttft, 0.99) * 1000, 3),
        "tpot_p50_ms": round(_at(tpot, 0.50) * 1000, 3),
        "decode_tok_s_p50": round(1.0 / _at(tpot, 0.50), 3),
        "decode_tok_s_p99": round(1.0 / _at(tpot, 0.99), 3),
        "e2e_p50_s": round(_at(e2e, 0.50), 3),
        "e2e_p99_s": round(_at(e2e, 0.99), 3),
    }
    _emit_summary(summary, args.summary_out)
    return summary


def run_bench(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m minisgl --bench",
        description="Fixed benchmark harness over the full serving stack.",
    )
    parser.add_argument("--model-path", required=True, help="local model dir or HF model id")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--dtype", default="auto", help="auto | float16 | bfloat16 | float32")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallelism size")
    parser.add_argument("--cache-type", default="radix", choices=["radix", "naive"])
    parser.add_argument("--attention-backend", default="auto", help="auto | fa | fi | trtllm")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=None)
    parser.add_argument("--max-running-requests", type=int, default=256)
    parser.add_argument("--memory-ratio", type=float, default=0.9)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--max-seq-len-override", type=int, default=None)
    parser.add_argument(
        "--suite",
        default="latency-1",
        choices=["latency-1", "mixed", "prefix-heavy"],
        help="workload shape; the replacement gate uses the suite named in config.toml [bench]",
    )
    parser.add_argument("--prompt-tokens", type=int, default=128, help="input length per request, in this tokenizer")
    parser.add_argument("--new-tokens", type=int, default=64, help="requested output length per request")
    parser.add_argument("--runs", type=int, default=5, help="measured requests")
    parser.add_argument("--warmup", type=int, default=2, help="warmup requests (dropped)")
    parser.add_argument(
        "--base-url",
        default=None,
        help="measure http://host:port that is already serving; do not start a server",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="also write the BENCH-SUMMARY json object to this path",
    )
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=600.0,
        help="max seconds to wait for a 1-token chat completion",
    )
    args = parser.parse_args(argv)

    if args.base_url:
        base = args.base_url.rstrip("/")
        logger.info(
            "bench client: base=%s suite=%s prompt_tokens=%d new_tokens=%d warmup=%d runs=%d",
            base,
            args.suite,
            args.prompt_tokens,
            args.new_tokens,
            args.warmup,
            args.runs,
        )
        model = _wait_serving(base, args.health_timeout)
        if not model:
            logger.error("server at %s did not complete a 1-token chat", base)
            return 1
        return 0 if _measure(base, model, args) else 1

    dtype = _resolve_dtype(args.model_path, args.dtype)
    server_args = ServerArgs(
        model_path=args.model_path,
        tp_info=DistributedInfo(0, args.tp),
        dtype=dtype,
        server_host=args.host,
        server_port=args.port,
        num_tokenizer=0,
        max_running_req=args.max_running_requests,
        attention_backend=args.attention_backend,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        page_size=args.page_size,
        memory_ratio=args.memory_ratio,
        cache_type=args.cache_type,
        max_seq_len_override=args.max_seq_len_override,
    )
    logger.info(
        "bench: model=%s suite=%s dtype=%s tp=%d prompt_tokens=%d new_tokens=%d warmup=%d runs=%d",
        args.model_path,
        args.suite,
        dtype,
        args.tp,
        args.prompt_tokens,
        args.new_tokens,
        args.warmup,
        args.runs,
    )

    from minisgl.server.api_server import app, start_frontend
    from minisgl.server.launch import start_subprocess

    # ---- bring up the full serving stack (identical to launch_server) ----
    start_frontend(server_args)
    start_subprocess(server_args, logger)
    logger.info("API server is ready to serve on %s:%s", args.host, args.port)

    # uvicorn in a daemon thread: off the main thread it skips installing
    # signal handlers (the bench drives the run loop and exits explicitly),
    # and all async work happens on its single event loop.
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True, name="minisgl-bench-uvicorn")
    thread.start()

    base = f"http://{args.host}:{args.port}"
    model = _wait_serving(base, args.health_timeout)
    if not model:
        logger.error("engine did not serve a 1-token chat within %.0f s", args.health_timeout)
        server.should_exit = True
        thread.join(timeout=10)
        _kill_worker_processes()
        return 1

    summary = _measure(base, model, args)

    # ---- teardown: stop uvicorn, close ZMQ, kill the backend workers ----
    server.should_exit = True
    thread.join(timeout=10)
    from minisgl.server.api_server import get_global_state

    try:
        get_global_state().shutdown()
    except AssertionError:
        pass
    _kill_worker_processes()

    return 0 if summary else 1


if __name__ == "__main__":
    sys.exit(run_bench())
