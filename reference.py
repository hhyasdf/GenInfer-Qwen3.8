#!/usr/bin/env python3
"""Independent float64 CPU reference — the source of the scaffold's golden file.

The golden test (``--selftest`` layer 2) compares the engine's prefill logits
and greedy tokens against this reference's output. The reference is
deliberately independent of the engine: it loads the same weights with
HuggingFace ``transformers`` at float64 on CPU and runs the canonical Llama
forward. No import from ``minisgl`` — independence is the point (a shared bug
would cancel out).

Float64 is the correctness oracle: the engine runs at bf16 (the toy model's
``torch_dtype``), and bf16 is exactly representable in f64, so both see identical
weight values and the only difference is floating-point accumulation — bounded
by the bf16 tolerance row in ``references/verification.md`` (max abs diff 1e-2).
The toy model's argmax margins are healthy (min top-1/top-2 gap ≫ the bf16
noise floor), so the greedy budget stays 0 (exact) rather than the bf16 row's
general 3 (10%).

The model-id canonicalization rule is duplicated here (not imported) for the
same independence reason: a local directory resolves to its absolute path; a
remote (HF) id is used as-is. The engine's self-test applies the identical rule,
so ``golden["model"]`` matches the engine's model id.

Usage:
  python reference.py --model-path models/tiny-llama \
      [--decode-steps 8] [--out golden/logits.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Short ASCII prompts (the toy tokenizer is byte-identity, so any ASCII text
# encodes to its byte values). Fixed set: they are part of the baked golden.
DEFAULT_PROMPTS = [
    "hello",
    "The quick brown fox",
    "2 + 2 =",
    "Once upon a time",
]

_GOLDEN_FORMAT = "gen-inference-golden/v1"


def canonical_model_id(model_path: str) -> str:
    """Local directory -> absolute resolved path; remote id -> as-is.

    Duplicated (not imported) from the engine's self-test for independence.
    """
    p = Path(model_path)
    return str(p.resolve()) if p.is_dir() else model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--out", default="golden/logits.json")
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float64
    )
    model.eval()
    model.to("cpu")
    tok = AutoTokenizer.from_pretrained(args.model_path)

    prompts_out = []
    min_gap = float("inf")
    for text in args.prompts:
        ids = tok.encode(text, add_special_tokens=False)
        input_ids = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1]  # last position
        # Greedy continuation: token 1 = argmax(prefill), then decode_steps-1 more.
        gen = [int(logits.argmax().item())]
        cur = input_ids
        for _ in range(args.decode_steps - 1):
            with torch.no_grad():
                out = model(cur).logits[0, -1]
            next_id = int(out.argmax().item())
            gen.append(next_id)
            cur = torch.cat([cur, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
        # Track the smallest top-1/top-2 logit gap (greedy stability estimate).
        with torch.no_grad():
            top2 = torch.topk(model(input_ids).logits[0, -1], 2).values
        min_gap = min(min_gap, float((top2[0] - top2[1]).item()))

        prompts_out.append(
            {
                "text": text,
                "ids": ids,
                "logits": logits.float().cpu().tolist(),
                "greedy_ids": gen,
                "greedy_max_mismatches": 0,
            }
        )
        print(f"  {text!r}: {len(ids)} input ids, {len(gen)} greedy ids")

    golden = {
        "format": _GOLDEN_FORMAT,
        "model": canonical_model_id(args.model_path),
        "dtype": "bf16",
        "tolerance": {"max_abs_diff": 1e-2},
        "prompts": prompts_out,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path} (model={golden['model']})")
    print(f"min top-1/top-2 logit gap: {min_gap:.6g} (greedy stability estimate)")


if __name__ == "__main__":
    main()
