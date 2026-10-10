"""CPU-only probe: does every GGUF tensor map to a model state_dict slot,
and do the shapes/dtypes line up? No CUDA, no kernels — just the reader,
the name map, and the model skeleton. Exits non-zero on any mismatch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch  # noqa: E402

from minisgl.layers.rotary import set_rope_device  # noqa: E402
from minisgl.models import create_model, load_weight  # noqa: E402
from minisgl.models.config import ModelConfig, RotaryConfig  # noqa: E402

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/hhy/Models/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-"
    "Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2"
)

# --- hyperparameters, cross-checked against the GGUF tensor inventory ---
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

# Build the model skeleton on the meta device: full state_dict key/shape/dtype
# info with zero RAM (the real skeleton is ~21.6 GiB; the host has ~7 GiB free
# because the origin service holds the GPUs). The engine's meta-build path
# materializes the rope table on the device set via set_rope_device().
set_rope_device(torch.device("cpu"))
with torch.device("meta"):
    model = create_model(config)
expected = {
    k: (tuple(v.shape), v.dtype) for k, v in model.state_dict().items()
}
print(f"model state_dict: {len(expected)} keys")

loaded = {}
n = 0
for key, t in load_weight(MODEL_DIR, torch.device("cpu")):
    loaded[key] = (tuple(t.shape), t.dtype)
    n += 1
    t = None
print(f"gguf loaded: {n} tensors")

missing = sorted(set(expected) - set(loaded))
unexpected = sorted(set(loaded) - set(expected))
mismatched = []
for k in sorted(set(expected) & set(loaded)):
    if expected[k] != loaded[k]:
        mismatched.append((k, expected[k], loaded[k]))

ok = True
if missing:
    ok = False
    print(f"\nMISSING from GGUF ({len(missing)}):")
    for k in missing:
        print(f"  {k}  expects {expected[k]}")
if unexpected:
    ok = False
    print(f"\nUNEXPECTED from GGUF ({len(unexpected)}):")
    for k in unexpected:
        print(f"  {k}  got {loaded[k]}")
if mismatched:
    ok = False
    print(f"\nSHAPE/DTYPE MISMATCH ({len(mismatched)}):")
    for k, e, g in mismatched[:20]:
        print(f"  {k}: expects {e}, got {g}")

print("\nPROBE PASS" if ok else "\nPROBE FAIL")
sys.exit(0 if ok else 1)
