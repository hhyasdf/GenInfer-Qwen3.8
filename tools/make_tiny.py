#!/usr/bin/env python3
"""Generate the scaffold's toy model: ``models/tiny-llama/``.

The toy model exists so the scaffold is self-testable without real weights:
``--selftest``, ``--bench``, and the HTTP round-trip all work out of the box.
The skill replaces it with the real model in steps 5-6.

Architecture (a minimal Llama the mini-sglang loader understands):
  hidden 128, 2 layers, 2 query heads, 1 KV head (GQA), head_dim 64,
  intermediate 384, vocab 255, tied embeddings, max_position 2048.
  head_dim must be in {64, 128, 256, 512} (flashinfer's RoPE kernel), so the
  smallest workable config is 2 qo heads x 64 = hidden 128.

Weights are stored as bfloat16 in ``model.safetensors`` (a few MB). The engine
runs at bf16 (config ``torch_dtype``) and the independent reference runs at
f64; bf16 is exactly representable in f64, so both see identical weight values
and the only difference is floating-point accumulation (the bf16 tolerance row).
bf16 (not f32) because the attention backends (flashinfer / flash-attention)
only dispatch fp16/bf16 — the same dtypes real models run at.

The tokenizer is a byte-identity BPE: token id == byte value, one token per
byte, so ids are interpretable ("hello" -> [104, 101, 108, 108, 111]) and the
golden test's ``encode(text) == ids`` check is deterministic. It is
deterministic and reversible, which is all the self-test and the benchmark's
prompt generator need.

Two details are load-bearing (the round-trip breaks without them):

* The vocab has **255** entries (bytes 0-254; the unreachable byte 255 is
  dropped), not 256. The benchmark's ``generate_prompt`` draws ids from
  ``[0, vocab_size // 2]`` *inclusively*; with 256 that range reaches id 128
  (0x80, a lone UTF-8 continuation byte) which decodes to U+FFFD and
  re-encodes to three bytes, so the length-adjustment loop oscillates. With
  255 the range is ``[0, 127]`` — all valid single-byte ASCII — and every
  prompt converges on the first attempt. The model's ``vocab_size`` matches
  (255), so every id the LM head can emit is decodable.
* ``tok.decoder = ByteLevel()`` is required. Without it ``decode`` joins
  tokens with spaces ("h e l l o") and the round-trip breaks. The ByteLevel
  decoder handles the ``Ġ`` -> space mapping plus char -> byte -> UTF-8.

The 243 reachable bytes map to the 256-char ByteLevel alphabet by
pre-tokenizing one-byte probe strings (the byte_encoder lives only in the
Rust binary, not as a Python dict); the 12 unreachable bytes (0xC0, 0xC1,
0xF5-0xFE) get private-use-area padding ``chr(0xE000 + b)``.

A minimal chat template is bundled so ``/v1/chat/completions`` (which calls
``apply_chat_template``) works.

Usage:  python tools/make_tiny.py [--out models/tiny-llama] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file


def build_config() -> dict:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "intermediate_size": 384,
        "vocab_size": 255,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-05,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.56.0",
    }


def build_weights(seed: int) -> dict[str, torch.Tensor]:
    """Random small projections, unit norms. All bfloat16, deterministic by seed."""
    g = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g, dtype=torch.bfloat16) * 0.02

    def ones(*shape: int) -> torch.Tensor:
        return torch.ones(*shape, dtype=torch.bfloat16)

    w: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": randn(255, 128),
    }
    for layer in range(2):
        p = f"model.layers.{layer}"
        w[f"{p}.self_attn.q_proj.weight"] = randn(128, 128)  # 2 qo heads * 64
        w[f"{p}.self_attn.k_proj.weight"] = randn(64, 128)  # 1 kv head * 64
        w[f"{p}.self_attn.v_proj.weight"] = randn(64, 128)
        w[f"{p}.self_attn.o_proj.weight"] = randn(128, 128)
        w[f"{p}.input_layernorm.weight"] = ones(128)
        w[f"{p}.post_attention_layernorm.weight"] = ones(128)
        w[f"{p}.mlp.gate_proj.weight"] = randn(384, 128)
        w[f"{p}.mlp.up_proj.weight"] = randn(384, 128)
        w[f"{p}.mlp.down_proj.weight"] = randn(128, 384)
    w["model.norm.weight"] = ones(128)
    # No lm_head.weight: embeddings are tied (the loader pops it when tied).
    return w


def _char_for_byte(pre, b: int) -> str | None:
    """The single ByteLevel-alphabet char that pre-tokenizes to byte ``b``.

    The byte_encoder is not exposed as a Python dict (it lives in the Rust
    binary), so it is derived by pre-tokenizing one-byte probe strings. For a
    byte that is the ``idx``-th byte of a valid UTF-8 encoding of code point
    ``cp``, the probe string is ``chr(cp)`` and the char is
    ``pre.pre_tokenize_str(chr(cp))[0][0][idx]``. Returns ``None`` for the
    bytes no valid UTF-8 sequence can produce (0xC0/0xC1 overlong 2-byte
    leads; 0xF5-0xFF invalid 4-byte leads).
    """
    if b == 32:  # space: the 'Ġ' char IS the byte-32 char, not a prefix
        return "Ġ"
    if b in (0xC0, 0xC1) or b >= 0xF5:
        return None
    if b < 0x80:
        cp, idx = b, 0
    elif b <= 0xBF:
        cp, idx = b, 1  # continuation byte: 2nd byte of a 2-byte sequence
    elif b <= 0xDF:
        cp, idx = (b - 0xC0) << 6, 0  # 2-byte lead
    elif b <= 0xEF:
        cp, idx = max(0x800, (b - 0xE0) << 12), 0  # 3-byte lead
    else:
        cp, idx = max(0x10000, (b - 0xF0) << 18), 0  # 4-byte lead
    s = chr(cp)
    enc = s.encode("utf-8")
    assert enc[idx] == b, f"constructed {s!r} encodes to {enc.hex()}, byte {idx}={enc[idx]} != {b}"
    word = pre.pre_tokenize_str(s)[0][0]
    assert not word.startswith("Ġ"), f"probe for byte {b} unexpectedly space-prefixed"
    return word[idx]


def build_tokenizer(out_dir: Path) -> None:
    """Byte-identity BPE: token id == byte value, one token per byte.

    255-entry vocab (bytes 0-254; unreachable byte 255 dropped) so the
    benchmark's inclusive ``[0, vocab_size // 2]`` id draw stays in ASCII.
    See the module docstring for why the 255 count and the ByteLevel decoder
    are both load-bearing.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    pre = pre_tokenizers.ByteLevel(add_prefix_space=False)
    vocab: dict[str, int] = {}
    for b in range(255):  # bytes 0..254; id == byte value
        ch = _char_for_byte(pre, b)
        if ch is None:  # unreachable byte: private-use-area padding
            ch = chr(0xE000 + b)
        assert ch not in vocab, f"byte {b} char {ch!r} collides"
        vocab[ch] = b
    assert len(vocab) == 255

    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre
    tok.decoder = decoders.ByteLevel()  # required: handles Ġ->space + char->byte->UTF-8
    tok.save(str(out_dir / "tokenizer.json"))

    chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": chat_template,
    }
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="models/tiny-llama")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps(build_config(), indent=2) + "\n", encoding="utf-8")
    weights = build_weights(args.seed)
    save_file(weights, str(out_dir / "model.safetensors"))
    build_tokenizer(out_dir)

    n_params = sum(t.numel() for t in weights.values())
    print(f"wrote {out_dir}/ ({len(weights)} tensors, {n_params} params, seed={args.seed})")
    for name in sorted(weights):
        print(f"  {name}: {list(weights[name].shape)}")


if __name__ == "__main__":
    main()
