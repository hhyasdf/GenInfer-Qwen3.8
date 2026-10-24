# D-6b — 真实 BPE tokenizer (qwen35 pre + GPT-2 byte fallback)。
#
# Round-1 用 hash 占位 tokenize (空格分词 → abs(hash)%vocab) → L1/L3 零有效 token。
# 现在逐字移植 llama.cpp:
#   - byte-escape 表: unicode.cpp unicode_byte_to_utf8 / unicode_utf8_to_byte。
#     0x21-0x7E / 0xA1-0xAC / 0xAE-0xFF 恒等; 其余 (0x00-0x20,0x7F-0xA0,0xAD) → U+0100+n。
#   - pre-tokenizer: unicode.cpp unicode_regex_split_custom_qwen35
#     (qwen35 正则: 撇号缩写 | 可选前缀+字母串(含变音符号) | 单数字 | 符号串 | 空白)。
#   - BPE merge: llama-vocab.cpp llm_tokenizer_bpe_session::tokenize
#     (按 merge rank 升序贪心合并, 字节级 symbol)。
#   - tokenize: text→pre-split→逐字 escape→BPE→text_to_token (字节回退)。
#   - detokenize: token→piece(escape 文本)→llama_decode_text (escape→byte→UTF-8)。
# 交叉验证: 与 llama-server /tokenize + /detokenize 对齐, detokenize(tokenize(x))==x 闭包。
from __future__ import annotations

import bisect


# ---------------------------------------------------------------------------
# byte-escape 表 (对照 llama.cpp unicode.cpp:148-189)
# ---------------------------------------------------------------------------
def _build_byte_escape():
    """byte(0-255) → escaped codepoint, 及其逆。逐字移植 unicode_byte_to_utf8_map。"""
    b2c = {}
    for ch in range(0x21, 0x7F):          # u'!' to u'~'
        b2c[ch] = ch
    for ch in range(0xA1, 0xAD):          # u'¡' to u'¬'
        b2c[ch] = ch
    for ch in range(0xAE, 0x100):         # u'®' to u'ÿ'
        b2c[ch] = ch
    n = 0
    for ch in range(256):
        if ch not in b2c:
            b2c[ch] = 256 + n
            n += 1
    c2b = {v: k for k, v in b2c.items()}
    return b2c, c2b


_B2C, _C2B = _build_byte_escape()


def byte_encode_text(text: str) -> str:
    """raw UTF-8 text → escaped codepoint text (token 列表 / merges 的存储形态)。
    对照 llama_utf8_to_byte + unicode_byte_to_utf8。"""
    out = []
    for byte in text.encode("utf-8"):
        out.append(chr(_B2C[byte]))
    return "".join(out)


def decode_escape_text(esc: str) -> str:
    """escaped codepoint text → raw UTF-8 (llama_decode_text 的逆)。"""
    out = bytearray()
    for ch in esc:
        cp = ord(ch)
        if cp in _C2B:
            out.append(_C2B[cp])
        else:
            # 非转义字符 (不应出现) — 原样透传 UTF-8
            out += ch.encode("utf-8")
    return out.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# unicode flags (对照 unicode_cpt_flags_from_cpt, 用 Python unicodedata 等价)
# ---------------------------------------------------------------------------
import unicodedata


def _flags(cp: int) -> tuple:
    """返回 (is_letter, is_number, is_accent_mark, is_whitespace)。"""
    cat = unicodedata.category(chr(cp))
    is_letter = cat.startswith("L")
    is_number = cat.startswith("N")
    is_accent_mark = cat.startswith("M")
    is_whitespace = cp in (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F,
                           0x20, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0xA0) \
        or cp in (0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007,
                  0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000)
    return is_letter, is_number, is_accent_mark, is_whitespace


def _tolower(cp: int) -> int:
    return ord(chr(cp).lower())


# ---------------------------------------------------------------------------
# qwen35 pre-tokenizer (对照 unicode.cpp:613-738 unicode_regex_split_custom_qwen35)
# 返回 word 的字节长度列表 (相对原始 UTF-8 文本的偏移长度)
# ---------------------------------------------------------------------------
def unicode_regex_split_qwen35(text: str) -> list[int]:
    """返回每个 word 的长度 (UTF-8 字节数)。逐字移植 C++ 自定义分词。"""
    cpts = [ord(c) for c in text]
    bpe_offsets = []
    _prev_end = 0

    def _add(end: int) -> int:
        nonlocal _prev_end
        assert _prev_end <= end <= len(cpts)
        ln = end - _prev_end
        if ln > 0:
            # 存 UTF-8 字节长度 (与 C++ offsets 一致: 后续按字节切)
            bpe_offsets.append(len(text[_prev_end:end].encode("utf-8")))
        _prev_end = end
        return ln

    pos = 0
    n = len(cpts)
    while pos < n:
        cpt = cpts[pos]
        f_letter, f_num, f_accent, f_ws = _flags(cpt)

        # (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if cpt == 0x27 and pos + 1 < n:
            nxt = _tolower(cpts[pos + 1])
            if nxt in (ord("s"), ord("t"), ord("m"), ord("d")):
                pos += _add(pos + 2)
                continue
            if pos + 2 < n:
                nxt2 = _tolower(cpts[pos + 2])
                if (nxt == ord("r") and nxt2 == ord("e")) or \
                   (nxt == ord("v") and nxt2 == ord("e")) or \
                   (nxt == ord("l") and nxt2 == ord("l")):
                    pos += _add(pos + 3)
                    continue

        # [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if cpt not in (0x0D, 0x0A) and not f_num:
            n_letter, n_num, n_accent, n_ws = _flags(cpts[pos + 1]) if pos + 1 < n else (False, False, False, False)
            if f_letter or f_accent or n_accent or n_letter:
                pos += 1
                while pos < n:
                    l, _, a, _ = _flags(cpts[pos])
                    if l or a:
                        pos += 1
                    else:
                        break
                _add(pos)
                continue

        # \p{N}  (单个数字)
        if f_num:
            pos += 1
            _add(pos)
            continue

        # ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        flags2 = _flags(cpts[pos + 1]) if (cpt == 0x20 and pos + 1 < n) else (f_letter, f_num, f_accent, f_ws)
        f2_letter, f2_num, f2_accent, f2_ws = flags2
        if not (f2_ws or f2_letter or f2_accent or f2_num) and (cpt != 0 or f_ws or f_letter or f_num or f_accent):
            if cpt == 0x20:
                pos += 1
            while pos < n:
                l2, n2, a2, w2 = _flags(cpts[pos])
                if w2 or l2 or a2 or n2:
                    break
                pos += 1
            while pos < n and cpts[pos] in (0x0D, 0x0A):
                pos += 1
            _add(pos)
            continue

        # 空白扫描
        num_ws = 0
        last_end_r_or_n = 0
        while pos + num_ws < n:
            _, _, _, w = _flags(cpts[pos + num_ws])
            if not w:
                break
            if cpts[pos + num_ws] in (0x0D, 0x0A):
                last_end_r_or_n = pos + num_ws + 1
            num_ws += 1

        # \s*[\r\n]+
        if last_end_r_or_n > 0:
            pos = last_end_r_or_n
            _add(pos)
            continue

        # \s+(?!\S)
        if num_ws > 1 and pos + num_ws < n:
            pos += num_ws - 1
            _add(pos)
            continue

        # \s+
        if num_ws > 0:
            pos += num_ws
            _add(pos)
            continue

        # no matches
        _add(pos + 1)
        pos += 1

    return bpe_offsets


# ---------------------------------------------------------------------------
# BPE tokenizer 主体
# ---------------------------------------------------------------------------
class Qwen35BPE:
    """qwen35 pre + GPT-2 byte BPE。tokens/merges 来自 GGUF 内嵌 tokenizer.ggml。"""

    def __init__(self, tokens: list[str], merges: list[str],
                 bos_token_id: int = -1, eos_token_id: int = -1):
        self.tokens = tokens
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        # token 文本 (escaped 形态) → id
        self._token_to_id = {t: i for i, t in enumerate(tokens)}
        # merges: 每条 "left right" → rank (数组下标)
        self._merge_rank = {}
        for i, m in enumerate(merges):
            self._merge_rank[m] = i

    def find_bpe_rank(self, left: str, right: str) -> int:
        return self._merge_rank.get(left + " " + right, -1)

    def _bpe_word(self, word_escaped: str) -> list[str]:
        """对单个 escaped word 做 BPE 合并, 返回 token 文本列表 (escaped)。"""
        # symbol 链: 每个字符一个 symbol (escaped 形态下的单字符)
        symbols = [{"text": c, "n": 1, "prev": i - 1, "next": (i + 1 if i + 1 < len(word_escaped) else -1)}
                   for i, c in enumerate(word_escaped)]
        # bigram 优先队列 (rank, counter, left, right, bigram_text)
        # 过期检查: left.text+right.text != bigram_text → 跳过 (对照 C++ bigram.text 校验)。
        import heapq
        heap = []
        counter = 0

        def add_bigram(left: int, right: int):
            nonlocal counter
            if left < 0 or right < 0 or right >= len(symbols):
                return
            ls, rs = symbols[left], symbols[right]
            if ls["n"] == 0 or rs["n"] == 0:
                return
            rank = self.find_bpe_rank(ls["text"], rs["text"])
            if rank < 0:
                return
            heapq.heappush(heap, (rank, counter, left, right, ls["text"] + rs["text"]))
            counter += 1

        for i in range(1, len(symbols)):
            add_bigram(i - 1, i)

        while heap:
            rank, _, left, right, btext = heapq.heappop(heap)
            ls, rs = symbols[left], symbols[right]
            if ls["n"] == 0 or rs["n"] == 0:
                continue
            if ls["text"] + rs["text"] != btext:
                continue  # 过期 bigram (一侧已并入更长符号)
            ls["n"] += rs["n"]
            rs["n"] = 0
            ls["text"] = ls["text"] + rs["text"]
            ls["next"] = rs["next"]
            if rs["next"] >= 0:
                symbols[rs["next"]]["prev"] = left
            add_bigram(ls["prev"], left)
            add_bigram(left, ls["next"])

        return [s["text"] for s in symbols if s["n"] > 0]

    def tokenize(self, text: str, add_special: bool = False) -> list[int]:
        """text → token ids。对照 llama-vocab.cpp llm_tokenizer_bpe_session::tokenize。"""
        output = []
        if add_special and self.bos_token_id is not None and 0 <= self.bos_token_id < len(self.tokens):
            output.append(self.bos_token_id)
        if not text:
            if add_special and self.eos_token_id is not None and 0 <= self.eos_token_id < len(self.tokens):
                output.append(self.eos_token_id)
            return output
        # pre-split → word 字节长度
        offsets = unicode_regex_split_qwen35(text)
        byte_len = len(text.encode("utf-8"))
        pos = 0
        for wl in offsets:
            word_bytes = text.encode("utf-8")[pos:pos + wl]
            pos += wl
            word = word_bytes.decode("utf-8")
            word_escaped = byte_encode_text(word)
            if self._token_to_id.get(word_escaped) is not None:
                # 整 word 已是 token (ignore_merges 路径)
                output.append(self._token_to_id[word_escaped])
                continue
            for piece in self._bpe_word(word_escaped):
                tid = self._token_to_id.get(piece)
                if tid is None:
                    # 字节回退: 逐 escaped 字符 → byte token
                    for ch in piece:
                        cp = ord(ch)
                        if cp in _C2B:
                            byte = _C2B[cp]
                            # byte token 的 escaped 文本 = chr(_B2C[byte])
                            bt = chr(_B2C[byte])
                            output.append(self._token_to_id[bt])
                else:
                    output.append(tid)
        if add_special and self.eos_token_id is not None and 0 <= self.eos_token_id < len(self.tokens):
            output.append(self.eos_token_id)
        return output

    def detokenize(self, token_ids, special: bool = False) -> str:
        """token ids → text。对照 llama_vocab::detokenize + llama_decode_text。"""
        pieces = []
        for tid in token_ids:
            if tid < 0 or tid >= len(self.tokens):
                continue
            attr = self._attr(tid)
            if not special and attr & (1 | 2):  # UNKNOWN|CONTROL → 抑制
                continue
            text = self.tokens[tid]
            if attr & 4:  # NORMAL
                pieces.append(decode_escape_text(text))
            elif attr & 8:  # BYTE
                b = _C2B.get(ord(text[0]), 0) if text else 0
                pieces.append(bytes([b]).decode("utf-8", errors="replace"))
            else:
                pieces.append(decode_escape_text(text))
        return "".join(pieces)

    def _attr(self, tid: int) -> int:
        """粗略 token 属性: 0=UNKNOWN,1=UNKNOWN,2=CONTROL,4=NORMAL,8=BYTE,16=SPECIAL。
        简化: 单字符且是 byte-escape 表内 → BYTE; 否则 NORMAL。特殊 token 由调用方判断。"""
        t = self.tokens[tid]
        if len(t) == 1 and ord(t[0]) in _C2B:
            return 8  # BYTE
        return 4      # NORMAL


def load_tokenizer(gguf_path) -> Qwen35BPE:
    """从 GGUF 内嵌 tokenizer.ggml 构建 Qwen35BPE。"""
    from .weights.gguf_loader import read_tokenizer_meta
    meta = read_tokenizer_meta(gguf_path)
    return Qwen35BPE(meta["tokens"], meta["merges"],
                     meta.get("bos_token_id", -1), meta.get("eos_token_id", -1))
