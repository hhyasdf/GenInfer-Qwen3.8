"""CPU-only diagnostic for the ssm_conv1d (GDN depthwise conv) weights.

Reads the GGUF with the engine's own GgufReader (no GPU, no full model load)
and reports, for every ssm_conv1d tensor:
  - the GGUF tensor type + declared shape
  - statistics when interpreted as float32 (min/max/mean/std/fraction-of-zeros)
  - whether the engine's expected shape [10240, 4] matches the GGUF shape

A conv kernel should be a small, dense, non-zero F32 tensor (a learned
depthwise causal conv). If it comes out all-zero, or its GGUF dtype is not
F32, or its shape is transposed vs [10240, 4], that is the ssm_conv1d bug.
"""
import sys
import torch

sys.path.insert(0, "python")
from minisgl.models.gguf_reader import GgufReader, GGML_F32, GGML_BF16, GGML_Q8_0, GGML_Q6_K  # noqa: E402
from minisgl.models.gguf_weight import find_main_gguf  # noqa: E402

model_path = "/home/hhy/Models/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2"
main = find_main_gguf(model_path)
print("main GGUF:", main)

dtype_name = {0: "F32", 1: "F16", 8: "Q8_0", 14: "Q6_K", 22: "BF16"}

with GgufReader(main) as r:
    convs = [t for t in r.tensors if "ssm_conv1d" in t.name]
    print(f"ssm_conv1d tensors found: {len(convs)}")
    for t in convs[:3]:
        raw = r.tensor_bytes(t)
        print(f"  name={t.name!r}  ggml_type={t.tensor_type} ({dtype_name.get(t.tensor_type, '?')})  shape={t.shape}  raw_bytes={len(raw)}")
        if t.tensor_type == GGML_F32:
            numel = 1
            for d in t.shape:
                numel *= d
            f = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(torch.float32)
            print(f"    numel={numel}  min={f.min().item():.6e}  max={f.max().item():.6e}  "
                  f"mean={f.mean().item():.6e}  std={f.std().item():.6e}  "
                  f"frac_zero={(f == 0).float().mean().item():.4f}")
            # Compare to the engine's expected [10240, 4] layout.
            if t.shape != [10240, 4]:
                print(f"    !! GGUF shape {t.shape} != engine-expected [10240, 4]")
                if f.numel() == 10240 * 4:
                    as_engine = f.view(10240, 4)
                    as_transposed = f.view(4, 10240)
                    print(f"    if reshaped to [10240,4]: min={as_engine.min().item():.6e} max={as_engine.max().item():.6e} mean={as_engine.mean().item():.6e}")
                    print(f"    if it is actually [4,10240]: min={as_transposed.min().item():.6e} max={as_transposed.max().item():.6e} mean={as_transposed.mean().item():.6e}")
        else:
            print(f"    !! expected F32 (type 0) but got type {t.tensor_type} — "
                  f"the engine's F32 reshape would misinterpret the bytes")

    # Cross-check other F32 scalars the GDN layer relies on.
    print("\n-- other GDN F32 tensors (blk.0) --")
    for t in r.tensors:
        if t.name.startswith("blk.0.") and any(s in t.name for s in ("ssm_dt", "ssm_a", "ssm_norm")):
            raw = r.tensor_bytes(t)
            numel = 1
            for d in t.shape:
                numel *= d
            f = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(torch.float32)
            print(f"  {t.name!r}  type={t.tensor_type} ({dtype_name.get(t.tensor_type, '?')})  shape={t.shape}  "
                  f"min={f.min().item():.6e} max={f.max().item():.6e} mean={f.mean().item():.6e} "
                  f"frac_zero={(f == 0).float().mean().item():.4f}")
