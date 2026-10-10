"""Re-capture engine prefill state with the current (fixed) code.

Same construction as bisect_prefill.py (single rank on cuda:1, the selftest's
exact build), but skips the dequantization (w/*.pt and small.pt are still
valid: the q6k dequant kernel and the GGUF reader are unchanged since they
were produced). Saves:
  /tmp/bisect/captures.pt        per-layer (h, r) + embed + final_hidden + logits
  /tmp/bisect/eng_l3_inter.pt    layer-3 intermediates (via QWEN35_DEBUG_L3)
and checks that GDN layers 0-2 are unchanged vs the previous captures
(captures_old.pt), which validates the existing f64 layer-3 reference.
"""
from __future__ import annotations

import json
import os
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # engine root (this script lives in hacks/)
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["QWEN35_DEBUG_L3"] = "1"

import torch

from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.config import SchedulerConfig
from minisgl.scheduler.scheduler import (
    _make_input_tuple,
    _make_positions,
    _make_write_tuple,
)
from minisgl.scheduler.table import TableManager
from minisgl.scheduler.utils import PendingReq
from minisgl.utils import init_logger

logger = init_logger("recapture")
OUT = Path("/tmp/bisect")
GPU = 1


def main() -> None:
    t0 = time.time()

    cfg = tomllib.load(open(ROOT / "config.toml", "rb"))
    model_path = cfg["engine"]["model"]
    dtype = getattr(torch, cfg["engine"].get("dtype", "bfloat16"))
    attention_backend = cfg["performance"].get("attention_backend", "auto")

    golden = json.loads((ROOT / "golden" / "logits.json").read_text())
    prompt = golden["prompts"][0]
    ids = torch.tensor(prompt["ids"], dtype=torch.int32)
    logger.info("prompt 0: %d tokens", ids.shape[0])

    config = SchedulerConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=dtype,
        cache_type="radix",
        cuda_graph_max_bs=0,
        device=f"cuda:{GPU}",
        num_page_override=256,
        attention_backend=attention_backend,
    )
    logger.info("building engine (model load ~67 s)...")
    engine = Engine(config)
    model = engine.model.model  # Qwen35Model

    # ---- capture hooks ----
    captures: dict[str, torch.Tensor] = {}

    embed = model.embed
    embed_orig = embed.forward

    def _embed_fwd(input_ids):
        out = embed_orig(input_ids)
        captures["embed"] = out.detach().cpu()
        return out

    embed.forward = _embed_fwd

    for i, layer in enumerate(model.layers):
        layer_orig = layer.forward

        def _layer_fwd(x, residual, _i=i, _orig=layer_orig):
            h, r = _orig(x, residual)
            captures[f"layer_{_i:02d}_h"] = h.detach().cpu()
            captures[f"layer_{_i:02d}_r"] = r.detach().cpu()
            return h, r

        layer.forward = _layer_fwd

    norm = model.norm
    norm_orig = norm.forward

    def _norm_fwd(h):
        out = norm_orig(h)
        captures["final_hidden"] = out.detach().cpu()
        return out

    norm.forward = _norm_fwd

    # ---- prefill (the selftest's prefill recipe) ----
    device = engine.device
    table_manager = TableManager(config.max_running_req, engine.page_table)
    cache_manager = CacheManager(
        engine.num_pages, config.page_size, engine.page_table, config.cache_type
    )
    token_pool = table_manager.token_pool

    sampling_params = SamplingParams(
        temperature=0.0, top_k=-1, top_p=1.0, ignore_eos=True, max_tokens=30
    )
    pending = PendingReq(uid=0, input_ids=ids, sampling_params=sampling_params)
    handle = cache_manager.match_req(pending).cuda_handle
    cached_len = handle.cached_len
    cache_manager.lock(handle)
    table_idx = table_manager.allocate()
    assert cached_len == 0, "expected a cold cache on a fresh engine"
    token_pool[table_idx, : ids.numel()].copy_(ids.pin_memory(), non_blocking=True)
    req = Req(
        input_ids=ids,
        table_idx=table_idx,
        cached_len=0,
        output_len=30,
        uid=0,
        sampling_params=sampling_params,
        cache_handle=handle,
    )

    batch = Batch(reqs=[req], phase="prefill")
    engine.graph_runner.pad_batch(batch)
    cache_manager.allocate_paged(batch.reqs)
    batch.positions = _make_positions(batch, device)
    input_mapping = _make_input_tuple(batch, device)
    write_mapping = _make_write_tuple(batch, device)
    batch.out_loc = engine.page_table[input_mapping]
    engine.attn_backend.prepare_metadata(batch)
    batch.input_ids = token_pool[input_mapping]
    with torch.cuda.stream(engine.stream), engine.ctx.forward_batch(batch):
        logits = engine.model.forward()
    # model.forward() returns one row per request (last position)
    logits = logits[0].float().cpu()
    captures["logits"] = logits
    logger.info(
        "prefill done: logits %s, max|logit|=%.3f (%.1f s)",
        tuple(logits.shape),
        logits.abs().max().item(),
        time.time() - t0,
    )

    # ---- save (preserve the old captures for the GDN-unchanged check) ----
    old = OUT / "captures_old.pt"
    cur = OUT / "captures.pt"
    if cur.exists() and not old.exists():
        cur.rename(old)
    torch.save(captures, cur)
    logger.info("captures saved (%d tensors)", len(captures))

    # ---- GDN-unchanged check: layers 0-2 must match the previous captures ----
    if old.exists():
        oldc = torch.load(old, map_location="cpu")
        for i in range(3):
            for suf in ("h", "r"):
                key = f"layer_{i:02d}_{suf}"
                a, b = oldc[key].float(), captures[key].float()
                d = (a - b).abs().max().item()
                logger.info("GDN check %s: max|diff|=%.3e", key, d)

    # ---- golden comparison (prefill logits + first greedy token) ----
    glog = torch.tensor(prompt["logits"], dtype=torch.float32)
    d = (logits - glog).abs()
    logger.info(
        "golden prefill logits: max|diff|=%.4f mean=%.5f (tol 0.01)",
        d.max().item(),
        d.mean().item(),
    )
    first = int(logits.argmax().item())
    logger.info(
        "first greedy token: engine=%d golden=%d %s",
        first,
        prompt["greedy_ids"][0],
        "MATCH" if first == prompt["greedy_ids"][0] else "MISMATCH",
    )
    logger.info("recapture done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
