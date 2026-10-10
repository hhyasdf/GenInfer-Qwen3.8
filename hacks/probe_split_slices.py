"""CPU-only verification of the layer-split weight path.

For each rank's slice [start, end), build the sliced model skeleton on the
meta device and walk the real GGUF weight stream, remapping global layer ids
to local ids exactly as engine._load_weight_state_dict does. PASS means every
rank's wanted key set is covered exactly once with zero MISSING / UNEXPECTED
/ MISMATCH (shape or dtype). No GPU required; the GGUF file is mmap'd on CPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch  # noqa: E402

from minisgl.engine.engine import _remap_layer_key  # noqa: E402
from minisgl.layers.rotary import set_rope_device  # noqa: E402
from minisgl.models import create_model, load_weight  # noqa: E402
from minisgl.models.config import ModelConfig, RotaryConfig  # noqa: E402

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/hhy/Models/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-"
    "Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2"
)
SPLIT = (26, 38)  # from config.toml [distributed].layer_split

# --- hyperparameters, cross-checked against the GGUF tensor inventory ---
# (identical to hacks/probe_weight_keys.py)
config = ModelConfig(
    num_layers=64,
    num_qo_heads=24,
    num_kv_heads=4,
    head_dim=256,
    hidden_size=5120,
    vocab_size=248320,
    intermediate_size=17408,
    rms_norm_eps=1e-6,
    rotary_config=RotaryConfig(
        head_dim=256, rotary_dim=256, max_position=102400, base=10000000.0, scaling=None
    ),
    hidden_act="silu",
    tie_word_embeddings=False,
    num_experts=0,
    num_experts_per_tok=0,
    moe_intermediate_size=0,
    norm_topk_prob=False,
    model_type="qwen35",
    architectures=["Qwen35ForCausalLM"],
    full_attn_layer_ids=frozenset(range(3, 64, 4)),
    gdn_conv_kernel=4,
    gdn_state_size=128,
    gdn_group_count=16,
    gdn_time_step_rank=48,
    gdn_inner_size=6144,
    mtp_enabled=True,
    mtp_n_max=4,
)

start = 0
failures = 0
covered_total = 0
for rank, chunk in enumerate(SPLIT):
    end = start + chunk
    set_rope_device(torch.device("cpu"))
    with torch.device("meta"):
        model = create_model(config, layer_range=(start, end))
    wanted = model.state_dict()
    got_keys = set()
    missing, mismatched = set(wanted), []
    dropped = 0  # loader keys this rank does not own (by design: the engine's
                 # ``wanted`` filter drops them; the other rank owns them)
    for name, t in load_weight(MODEL_DIR, torch.device("cpu")):
        rk = _remap_layer_key(name, start, end)
        if rk is None:
            continue
        if rk not in wanted:
            dropped += 1
            continue
        w = wanted[rk]
        if tuple(w.shape) != tuple(t.shape) or w.dtype != t.dtype:
            mismatched.append((name, rk, tuple(w.shape), tuple(t.shape), w.dtype, t.dtype))
        got_keys.add(rk)
        missing.discard(rk)
    print(f"rank {rank} (layers {start}..{end - 1}, local 0..{chunk - 1}):")
    print(f"  wanted={len(wanted)} covered={len(got_keys)} missing={len(missing)} "
          f"dropped_other_rank={dropped} mismatched={len(mismatched)}")
    if missing or mismatched:
        failures += 1
        for k in sorted(missing)[:10]:
            print(f"  MISSING  {k}")
        for m in mismatched[:10]:
            print(f"  MISMATCH {m}")
    start = end

print("PROBE PASS — layer-split weight path verified on both ranks" if not failures
      else f"PROBE FAIL — {failures} rank(s) with problems")
sys.exit(0 if not failures else 1)
