"""GGUF BPE vocab loader.

Ports the llama.cpp BPE tokenizer (gpt2 model type, qwen35 pre-tokenizer)
from GGUF metadata into an HF-compatible tokenizer interface.

The GGUF stores:
  - tokenizer.ggml.tokens: list[248320] of byte-level strings (indexed by token id)
  - tokenizer.ggml.merges: list[247587] of byte-level strings (each "a b" = merge of a+b)
  - tokenizer.ggml.token_type: list[248320] of token types (1=normal, 2=unknown, 3=control, 6=byte)
  - tokenizer.ggml.bos_token_id / eos / padding / mask
  - tokenizer.ggml.add_bos_token: bool
  - tokenizer.chat_template: jinja template string

Byte-level encoding:
  Both the tokens and the merges are stored as *byte-level strings* — each
  original byte is represented by its gpt2 fallback character (e.g. space 0x20
  -> 'Ġ', 0xA1 -> '¡', and the 3 bytes of '你' (E4 BD A0) -> 'ä½ł'). This is the
  same encoding the `tokenizers` ByteLevel pre-tokenizer produces, so the BPE
  vocab is simply {token: id} over the raw GGUF strings and the merges are used
  as-is (no re-encoding needed).

The BPE algorithm (via the `tokenizers` library):
  1. Pre-tokenize the text into words with the qwen35 algorithm (a faithful
     Python port of llama.cpp's unicode_regex_split_custom_qwen35, which
     operates on Unicode codepoints with per-codepoint flags). Each word is
     byte-encoded into a byte-level string.
  2. Apply the merges in rank order to each byte-level string.
  3. Map each resulting byte-level string to its GGUF token id.

Decode inverts the byte-level encoding: byte-level string -> bytes -> UTF-8.
"""

from __future__ import annotations

import unicodedata
from typing import Any, List, Optional, Sequence

import torch


def _build_byte_map() -> dict[int, str]:
    """Build the gpt2 byte->fallback-char map (243 valid-UTF-8 bytes).

    The 13 missing bytes (0xC0, 0xC1, 0xF5-0xFF) are invalid UTF-8 lead bytes
    and never appear in valid text. Each byte maps to a single fallback char.
    """
    from tokenizers import pre_tokenizers

    bl = pre_tokenizers.ByteLevel(use_regex=False, add_prefix_space=False)
    byte_to_char: dict[int, str] = {}
    # b < 128: chr(b) is a single byte in UTF-8, so pre_tokenize_str gives
    # the fallback char directly (space -> 'Ġ', NUL -> 'Ā', etc.).
    for b in range(128):
        enc = bl.pre_tokenize_str(chr(b))[0][0]
        assert len(enc) == 1, f"byte {b:#04x} -> {enc!r} not a single char"
        byte_to_char[b] = enc
    # b >= 128: chr(b) is not a single byte, so iterate codepoints and zip
    # their UTF-8 bytes with the byte-level fallback chars.
    for cp in range(0x80, 0x110000):
        if len(byte_to_char) == 256:
            break
        if 0xD800 <= cp <= 0xDFFF:
            continue
        s = chr(cp)
        enc = bl.pre_tokenize_str(s)[0][0]
        utf8 = s.encode("utf-8")
        for b, ch in zip(utf8, enc):
            if b not in byte_to_char:
                byte_to_char[b] = ch
    return byte_to_char


class GGUFBPETokenizer:
    """HF-compatible BPE tokenizer loaded from GGUF metadata.

    Implements the minimal interface required by TokenizeManager:
      - encode(text, return_tensors="pt") -> torch.Tensor [1, n]
      - decode(ids) -> str
      - apply_chat_template(messages, tokenize=False, add_generation_prompt=True) -> str
    """

    def __init__(self, metadata: dict[str, Any]) -> None:
        self._tokens: List[str] = metadata["tokenizer.ggml.tokens"]
        self._merges_raw: List[str] = metadata["tokenizer.ggml.merges"]
        self._token_type: List[int] = metadata["tokenizer.ggml.token_type"]
        self.bos_id: int = metadata["tokenizer.ggml.bos_token_id"]
        self.eos_id: int = metadata["tokenizer.ggml.eos_token_id"]
        self.padding_id: int = metadata["tokenizer.ggml.padding_token_id"]
        self.mask_id: int = metadata.get("tokenizer.ggml.mask_token_id", -1)
        self.add_bos: bool = metadata["tokenizer.ggml.add_bos_token"]
        self.chat_template: str = metadata["tokenizer.chat_template"]

        # token (byte-level string) -> GGUF token id
        self._token_to_id: dict[str, int] = {}
        for i, tok in enumerate(self._tokens):
            self._token_to_id[tok] = i

        # Parse merges: each merge string "a b" is split at the first space
        # after position 0 (llama.cpp convention). Both parts are byte-level
        # strings, used as-is.
        self._merges: List[tuple[str, str]] = []
        for word in self._merges_raw:
            pos = word.find(" ", 1)
            if pos != -1:
                first = word[:pos]
                second = word[pos + 1 :]
                self._merges.append((first, second))

        # Byte maps: byte -> fallback char (for encode) and fallback char -> byte (for decode).
        byte_map = _build_byte_map()
        self._byte_to_char: dict[int, str] = byte_map
        self._char_to_byte: dict[str, int] = {ch: b for b, ch in byte_map.items()}

        # Build the tokenizers BPE
        self._build_bpe()

        # Special-token cache: tokens with type UNKNOWN(2), CONTROL(3), or
        # USER_DEFINED(4). These are isolated from raw text before BPE
        # (llama.cpp's tokenizer_st_partition). Sorted by byte-length
        # descending so longer tokens match first (greedy longest-match).
        self._special_tokens: List[tuple[str, int]] = []
        for i, tok in enumerate(self._tokens):
            if self._token_type[i] in (2, 3, 4):
                unicode_text = self._byte_decode(tok)
                if unicode_text:
                    self._special_tokens.append((unicode_text, i))
        self._special_tokens.sort(key=lambda x: len(x[0]), reverse=True)

    def _build_bpe(self) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE

        # Tokens and merges are already byte-level strings, so the vocab is
        # {token: id} over the raw GGUF strings and the merges are used as-is.
        vocab = {tok: i for i, tok in enumerate(self._tokens)}
        bpe = BPE(vocab, self._merges, dropout=None, unk_token=None)
        # No pre-tokenizer: we feed pre-tokenized byte-level words directly
        # (see _qwen35_pretokenize). The BPE model treats each input as a
        # single word and applies the merges in rank order.
        self._tokenizer = Tokenizer(bpe)

    def _byte_encode(self, text: str) -> str:
        """Convert a substring to a byte-level string (gpt2 fallback chars)."""
        return "".join(self._byte_to_char[b] for b in text.encode("utf-8"))

    def _byte_decode(self, byte_level_str: str) -> str:
        """Convert a byte-level string (gpt2 fallback chars) back to Unicode."""
        raw = bytearray()
        for ch in byte_level_str:
            b = self._char_to_byte.get(ch)
            if b is None:
                return ""
            raw.append(b)
        return raw.decode("utf-8", errors="replace")

    def _partition_special(self, text: str) -> List[tuple[str, Optional[int]]]:
        """Split text into segments: (raw_text, None) and (special_text, token_id).

        Port of llama.cpp's tokenizer_st_partition (llama-vocab.cpp:3252).
        Greedy longest-match scan: at each position, try the longest special
        token first; if it matches, emit it as a token segment and advance.
        Otherwise accumulate raw text until the next special-token match.
        """
        if not text:
            return []
        special_tokens = self._special_tokens
        segments: List[tuple[str, Optional[int]]] = []
        pos = 0
        n = len(text)
        while pos < n:
            # Try to match a special token at the current position (longest first)
            matched = False
            for unicode_text, token_id in special_tokens:
                if text.startswith(unicode_text, pos):
                    segments.append((unicode_text, token_id))
                    pos += len(unicode_text)
                    matched = True
                    break
            if not matched:
                # Find the earliest position where any special token matches
                next_pos = n
                for unicode_text, _tid in special_tokens:
                    idx = text.find(unicode_text, pos + 1)
                    if idx != -1 and idx < next_pos:
                        next_pos = idx
                segments.append((text[pos:next_pos], None))
                pos = next_pos
        return segments

    def _qwen35_pretokenize(self, text: str) -> List[str]:
        """Port of llama.cpp's unicode_regex_split_custom_qwen35 (src/unicode.cpp:610).

        Operates on Unicode codepoints (not byte-level strings) with per-codepoint
        flags. Returns a list of byte-level strings (gpt2 fallback chars), one per
        word. The word boundaries match llama.cpp's qwen35 pre-tokenizer exactly.

        The 8 alternatives (in priority order):
          (a) contractions: (?i:'s|'t|'re|'ve|'m|'ll|'d)
          (b) [^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+  (letter run, optional leading symbol)
          (c) \\p{N}  (single number)
          (d) <space>?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*  (symbol run, optional leading space)
          (e) \\s*[\\r\\n]+  (whitespace + newline)
          (f) \\s+(?!\\S)  (whitespace not followed by non-space)
          (g) \\s+  (whitespace)
          (h) fallback: single char
        """
        cpts = [ord(c) for c in text]
        n = len(cpts)
        if n == 0:
            return []

        # Precompute flags: (is_letter, is_number, is_accent_mark, is_whitespace)
        fl: List[tuple[bool, bool, bool, bool]] = []
        for cp in cpts:
            cat = unicodedata.category(chr(cp))
            fl.append((cat[0] == "L", cat[0] == "N", cat[0] == "M", chr(cp).isspace()))

        def get_cpt(pos: int) -> Optional[int]:
            return cpts[pos] if 0 <= pos < n else None

        def get_flags(pos: int) -> tuple[bool, bool, bool, bool]:
            return fl[pos] if 0 <= pos < n else (False, False, False, False)

        words: List[tuple[int, int]] = []  # (start, end) codepoint ranges
        prev_end = 0

        def add_token(end: int) -> None:
            nonlocal prev_end
            if end > prev_end:
                words.append((prev_end, end))
            prev_end = end

        pos = 0
        while pos < n:
            cpt = cpts[pos]
            is_letter, is_number, is_accent, _is_whitespace = get_flags(pos)

            # (a) contractions: (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if cpt == 0x27 and pos + 1 < n:  # '
                next_cp = get_cpt(pos + 1)
                next_lower = chr(next_cp).lower() if next_cp is not None else ""
                if next_lower in ("s", "t", "m", "d"):
                    add_token(pos + 2)
                    pos += 2
                    continue
                if pos + 2 < n:
                    next2_cp = get_cpt(pos + 2)
                    next2_lower = chr(next2_cp).lower() if next2_cp is not None else ""
                    if (next_lower == "r" and next2_lower == "e") or (
                        next_lower == "v" and next2_lower == "e"
                    ) or (next_lower == "l" and next2_lower == "l"):
                        add_token(pos + 3)
                        pos += 3
                        continue

            # (b) [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
            # Matches a letter run, optionally preceded by a single non-letter symbol.
            # Key: if the NEXT char is a letter, the current symbol is included
            # in the word (so ".com", "(x", "\tt" become single words).
            if cpt != 0x0D and cpt != 0x0A and not is_number:
                next_flags = get_flags(pos + 1)
                if is_letter or is_accent or next_flags[2] or next_flags[0]:
                    pos += 1
                    while get_flags(pos)[0] or get_flags(pos)[2]:
                        pos += 1
                    add_token(pos)
                    continue

            # (c) \p{N}
            if is_number:
                pos += 1
                add_token(pos)
                continue

            # (d) <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
            # Matches a symbol run, optionally preceded by a single space.
            flags2 = get_flags(pos + 1) if cpt == 0x20 else get_flags(pos)
            f2_ws, f2_letter, f2_accent, f2_number = flags2
            if not (f2_ws or f2_letter or f2_accent or f2_number):
                if cpt == 0x20:
                    pos += 1
                while pos < n:
                    f2_ws, f2_letter, f2_accent, f2_number = get_flags(pos)
                    if f2_ws or f2_letter or f2_accent or f2_number:
                        break
                    pos += 1
                # Consume trailing \r\n
                while pos < n and cpts[pos] in (0x0D, 0x0A):
                    pos += 1
                add_token(pos)
                continue

            # Count consecutive whitespace
            num_ws = 0
            last_end_rn = 0
            while get_flags(pos + num_ws)[3]:
                cpt2 = get_cpt(pos + num_ws)
                if cpt2 in (0x0D, 0x0A):
                    last_end_rn = pos + num_ws + 1
                num_ws += 1

            # (e) \s*[\r\n]+
            if last_end_rn > 0:
                pos = last_end_rn
                add_token(pos)
                continue

            # (f) \s+(?!\S)
            if num_ws > 1 and get_cpt(pos + num_ws) is not None:
                pos += num_ws - 1
                add_token(pos)
                continue

            # (g) \s+
            if num_ws > 0:
                pos += num_ws
                add_token(pos)
                continue

            # (h) fallback: single char
            pos += 1
            add_token(pos)

        # Convert codepoint ranges to byte-level strings
        return [self._byte_encode(text[s:e]) for s, e in words]

    # ------------------------------------------------------------------
    # HF-compatible interface
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_tensors: Optional[str] = None,
    ) -> List[int] | torch.Tensor:
        """Tokenize text into a list of token ids (or a torch.Tensor).

        Two-phase tokenization (matching llama.cpp's BPE tokenize path):
          1. tokenizer_st_partition: isolate special tokens (type 2/3/4) from
             raw text via greedy longest-match substring scan. Special tokens
             are emitted as single ids; only the remainders go through BPE.
          2. For each raw-text remainder: pre-tokenize with the qwen35
             algorithm, then run BPE on each byte-level word. Words already
             in the vocab are emitted directly (ignore_merges short-circuit).
        """
        ids: List[int] = []
        for segment_text, special_id in self._partition_special(text):
            if special_id is not None:
                # Special token: emit directly as a single id
                ids.append(special_id)
            else:
                # Raw text: pre-tokenize + BPE
                if not segment_text:
                    continue
                byte_words = self._qwen35_pretokenize(segment_text)
                for word in byte_words:
                    if not word:
                        continue
                    if word in self._token_to_id:
                        ids.append(self._token_to_id[word])
                    else:
                        enc = self._tokenizer.encode(word)
                        ids.extend(self._token_to_id[t] for t in enc.tokens)
        if add_special_tokens and self.add_bos:
            ids = [self.bos_id] + ids
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.int32)
        return ids

    def decode(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode a list of token ids into a string.

        Each token is a byte-level string; convert it back to bytes and UTF-8
        decode. Tokens are concatenated at the byte level before decoding so
        that multi-byte UTF-8 sequences split across tokens are reassembled.
        """
        raw = bytearray()
        for i in ids:
            if skip_special_tokens and self._is_special(i):
                continue
            for ch in self._tokens[i]:
                raw.append(self._char_to_byte[ch])
        return raw.decode("utf-8", errors="replace")

    def apply_chat_template(
        self,
        messages: List[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str | List[int]:
        """Render the jinja chat template."""
        from jinja2 import Environment, meta

        env = Environment()
        template = env.from_string(self.chat_template)
        prompt = template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        if tokenize:
            return self.encode(prompt)
        return prompt

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_special(self, token_id: int) -> bool:
        """Check if a token is special (control or user-defined type)."""
        if token_id < 0 or token_id >= len(self._token_type):
            return False
        # token_type: 1=normal, 2=unknown, 3=control, 4=user_defined, 5=byte
        return self._token_type[token_id] in (2, 3, 4)

    @property
    def vocab_size(self) -> int:
        return len(self._tokens)

    @property
    def model_max_length(self) -> int:
        return 100000  # engine max context

    @property
    def eos_token_id(self) -> int:
        return self.eos_id

    @property
    def bos_token_id(self) -> int:
        return self.bos_id

    def batch_decode(
        self,
        ids: Sequence[Sequence[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode a batch of token-id sequences into strings."""
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in ids]

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        **kwargs: Any,
    ) -> List[int] | torch.Tensor:
        return self.encode(text, return_tensors=return_tensors, **kwargs)
