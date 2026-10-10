"""Verify the _build_byte_map fix on CPU.

Replicates the bench's exact failure path: load_tokenizer(GGUF dir) then
generate_prompt(tokenizer, 60000) with random.seed(0). Before the fix this
raised KeyError('û') — a 60k random-token decode hit a token whose byte-level
string held the fallback char for invalid byte 0xFB ('û').
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from minisgl.tokenizer.gguf_vocab import _build_byte_map  # noqa: E402
from minisgl.utils import load_tokenizer  # noqa: E402

MODEL_DIR = "/home/hhy/Models/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2"


def generate_prompt(tokenizer, n: int) -> str:
    """Verbatim copy of benchmark/client.py:generate_prompt."""
    vocab_size = tokenizer.vocab_size // 2
    token_ids = [random.randint(0, vocab_size) for _ in range(n)]
    for it in range(64):
        prompt = tokenizer.decode(token_ids)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) == n:
            return prompt, it + 1, len(prompt)
        if len(token_ids) < n:
            need = n - len(token_ids)
            token_ids.extend([random.randint(0, vocab_size) for _ in range(need)])
        else:
            token_ids = token_ids[:n]
    raise ValueError("Failed to generate a message of the desired length.")


def main() -> None:
    # 1) byte map must cover all 256 bytes
    bm = _build_byte_map()
    assert len(bm) == 256, f"byte map has {len(bm)} entries, expected 256"
    assert bm[0xFB] == "\u00fb", f"byte 0xFB -> {bm[0xFB]!r}, expected '\u00fb' (û)"
    assert bm[0xC0] and bm[0xC1] and bm[0xF5] and bm[0xFF]
    print(f"[ok] byte map: {len(bm)} entries (all 256), 0xFB -> {bm[0xFB]!r}")

    # 2) load the engine tokenizer from the GGUF dir
    tok = load_tokenizer(MODEL_DIR)
    c2b = tok._char_to_byte
    assert len(c2b) == 256, f"_char_to_byte has {len(c2b)} entries, expected 256"
    assert c2b["\u00fb"] == 0xFB, f"_char_to_byte['û'] = {c2b.get(chr(0xFB))}, expected 0xFB"
    print(f"[ok] tokenizer vocab_size={tok.vocab_size}, _char_to_byte={len(c2b)} entries, 'û'->0xFB")

    # 3) the exact bench failure path
    random.seed(0)
    prompt, iters, chars = generate_prompt(tok, 60000)
    assert tok.encode(prompt, add_special_tokens=False) is not None
    n = len(tok.encode(prompt, add_special_tokens=False))
    assert n == 60000, f"prompt re-encodes to {n} tokens, expected 60000"
    print(f"[ok] generate_prompt(60000) succeeded after {iters} iteration(s), {chars} chars, re-encodes to {n} tokens")

    # 4) ASCII + multibyte round-trip unchanged
    for text in ("hello world", "hello world 你好", "preserve_thinking"):
        ids = tok.encode(text, add_special_tokens=False)
        back = tok.decode(ids)
        assert tok.encode(back, add_special_tokens=False) == ids, f"round-trip changed for {text!r}"
    print("[ok] ASCII + multibyte encode/decode round-trip stable")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
