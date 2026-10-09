"""B4 bisection — GPU side (run as a supervisor gpu-job).

Loads the engine single-rank on cuda:1 (exactly the selftest's construction),
runs the prefill of golden prompt 0, and captures:
  - the embedding output,
  - the (h, residual) state after each of the 64 layers,
  - the final hidden state (after the final norm),
  - the f32 logits at the last position.
Then dequantizes every Q6_K weight (via the engine's q6k_gather kernel) to
bf16 chunk files under /tmp/bisect/w/ for the f64 CPU reference, plus all
small (non-quantized) weights in /tmp/bisect/small.pt.

Also saves /tmp/bisect/reader_geom.json (the engine reader's view of the
GGUF tensor geometry) and /tmp/bisect/meta.json for the CPU side.

Dequant buffers are chunked (CHUNK rows at a time): the 20.8 GB model already
fills cuda:1, so a one-shot 2.5 GB buffer for embed/lm_head would OOM.
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

logger = init_logger("bisect")

OUT = Path("/tmp/bisect")
W = OUT / "w"
GPU = 1
CHUNK = 16384  # rows per dequant chunk (168 MB bf16 for K=5120)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    W.mkdir(parents=True, exist_ok=True)

    cfg = tomllib.load(open(ROOT / "config.toml", "rb"))
    model_path = cfg["engine"]["model"]
    dtype = getattr(torch, cfg["engine"].get("dtype", "bfloat16"))
    attention_backend = cfg["performance"].get("attention_backend", "auto")

    golden = json.loads((ROOT / "golden" / "logits.json").read_text())
    prompt = golden["prompts"][0]
    ids = torch.tensor(prompt["ids"], dtype=torch.int32)
    logger.info("prompt 0: %d tokens", ids.shape[0])

    # ---- engine (the selftest's exact construction) ----
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

    # ---- reader geometry (for the CPU-side independent parser cross-check) ----
    from minisgl.models.gguf_reader import GgufReader
    from minisgl.models.gguf_weight import find_main_gguf

    with GgufReader(find_main_gguf(model_path)) as reader:
        geom = {
            info.name: {
                "shape": list(info.shape),
                "type": int(info.tensor_type),
                "offset": int(info.offset),
            }
            for info in reader.tensors
        }
    (OUT / "reader_geom.json").write_text(json.dumps(geom))
    logger.info("reader geometry: %d tensors", len(geom))

    # ---- capture hooks (BaseOP is not an nn.Module: monkeypatch forward) ----
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
    logits = logits[0].float().cpu()
    captures["logits"] = logits
    logger.info(
        "prefill done: logits %s, max|logit|=%.3f (%.1f s)",
        tuple(logits.shape),
        logits.abs().max().item(),
        time.time() - t0,
    )

    torch.save(captures, OUT / "captures.pt")
    logger.info("captures saved (%d tensors)", len(captures))

    # ---- dequantize every Q6_K weight to bf16 chunk files ----
    from minisgl.kernel import q6k

    def dequant(weight: torch.Tensor, name: str) -> None:
        N = weight.shape[0]
        K = weight.shape[1] // 210 * 256
        for c in range(0, N, CHUNK):
            e = min(c + CHUNK, N)
            idx = torch.arange(c, e, dtype=torch.int32, device=weight.device)
            y = torch.empty(e - c, K, dtype=torch.bfloat16, device=weight.device)
            q6k.q6k_gather(idx, weight, y)
            torch.save(y.cpu(), W / f"{name}_c{c // CHUNK}.pt")
        logger.info("dequant %-24s [%d, %d] (%.1f s)", name, N, K, time.time() - t0)

    dequant(model.embed.weight, "embed")
    dequant(engine.model.lm_head.weight, "lm_head")
    for i, layer in enumerate(model.layers):
        if layer.is_full_attn:
            a = layer.attn
            dequant(a.attn_q.weight, f"L{i:02d}_attn_q")
            dequant(a.attn_k.weight, f"L{i:02d}_attn_k")
            dequant(a.attn_v.weight, f"L{i:02d}_attn_v")
            dequant(a.attn_output.weight, f"L{i:02d}_attn_output")
        else:
            a = layer.attn
            dequant(a.attn_qkv.weight, f"L{i:02d}_attn_qkv")
            dequant(a.attn_gate.weight, f"L{i:02d}_attn_gate")
            dequant(a.ssm_beta.weight, f"L{i:02d}_ssm_beta")
            dequant(a.ssm_alpha.weight, f"L{i:02d}_ssm_alpha")
            dequant(a.ssm_out.weight, f"L{i:02d}_ssm_out")
        dequant(layer.mlp.gate.weight, f"L{i:02d}_mlp_gate")
        dequant(layer.mlp.up.weight, f"L{i:02d}_mlp_up")
        dequant(layer.mlp.down.weight, f"L{i:02d}_mlp_down")

    # ---- small (non-quantized) weights, as-is (bf16) ----
    small = {"final_norm": model.norm.weight.detach().cpu()}
    for i, layer in enumerate(model.layers):
        d = {
            "input_layernorm": layer.input_layernorm.weight.detach().cpu(),
            "post_attention_layernorm": layer.post_attention_layernorm.weight.detach().cpu(),
        }
        if layer.is_full_attn:
            d["attn_q_norm"] = layer.attn.attn_q_norm.weight.detach().cpu()
            d["attn_k_norm"] = layer.attn.attn_k_norm.weight.detach().cpu()
        else:
            a = layer.attn
            d["ssm_conv1d"] = a.ssm_conv1d.detach().cpu()
            d["ssm_dt"] = a.ssm_dt.detach().cpu()
            d["ssm_a"] = a.ssm_a.detach().cpu()
            d["ssm_norm"] = a.ssm_norm.weight.detach().cpu()
        small[f"L{i:02d}"] = d
    torch.save(small, OUT / "small.pt")

    # ---- meta ----
    meta = {
        "prompt_ids": prompt["ids"],
        "layer_kinds": [
            "full_attn" if layer.is_full_attn else "gdn" for layer in model.layers
        ],
        "eps": float(config.model_config.rms_norm_eps),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("bisect_prefill done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
