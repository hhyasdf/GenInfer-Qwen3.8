"""Minimal: run engine prefill, dump layer-3 full-attn intermediates."""
from __future__ import annotations
import json, os, sys, time, tomllib
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
from minisgl.scheduler.scheduler import _make_input_tuple, _make_positions, _make_write_tuple
from minisgl.scheduler.table import TableManager
from minisgl.scheduler.utils import PendingReq

def main():
    t0 = time.time()
    cfg = tomllib.load(open(ROOT / "config.toml", "rb"))
    model_path = cfg["engine"]["model"]
    dtype = getattr(torch, cfg["engine"].get("dtype", "bfloat16"))
    attention_backend = cfg["performance"].get("attention_backend", "auto")

    golden = json.loads((ROOT / "golden" / "logits.json").read_text())
    prompt = golden["prompts"][0]
    ids = torch.tensor(prompt["ids"], dtype=torch.int32)
    print(f"prompt 0: {ids.shape[0]} tokens", flush=True)

    config = SchedulerConfig(
        model_path=model_path, tp_info=DistributedInfo(0, 1),
        dtype=dtype, cache_type="radix", cuda_graph_max_bs=0,
        device=f"cuda:{1}", num_page_override=256, attention_backend=attention_backend,
    )
    print("building engine...", flush=True)
    engine = Engine(config)
    model = engine.model.model

    device = engine.device
    table_manager = TableManager(config.max_running_req, engine.page_table)
    cache_manager = CacheManager(engine.num_pages, config.page_size, engine.page_table, config.cache_type)
    token_pool = table_manager.token_pool

    sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0, ignore_eos=True, max_tokens=30)
    pending = PendingReq(uid=0, input_ids=ids, sampling_params=sampling_params)
    handle = cache_manager.match_req(pending).cuda_handle
    cached_len = handle.cached_len
    cache_manager.lock(handle)
    table_idx = table_manager.allocate()
    assert cached_len == 0
    token_pool[table_idx, : ids.numel()].copy_(ids.pin_memory(), non_blocking=True)
    req = Req(input_ids=ids, table_idx=table_idx, cached_len=0, output_len=30, uid=0,
              sampling_params=sampling_params, cache_handle=handle)

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
    print(f"prefill done: logits {tuple(logits.shape)} ({time.time()-t0:.1f} s)", flush=True)

    # Also dump the x input to layer 3 (the normalized residual)
    # The debug hook already dumps x, qg, q, k, v, gate, o, positions
    print(f"dumped layer-3 intermediates to /tmp/bisect/eng_l3_inter.pt", flush=True)
    print(f"total {time.time()-t0:.1f} s", flush=True)

if __name__ == "__main__":
    main()
