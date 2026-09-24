"""Oracle comparison test for GGUFBPETokenizer."""
import sys, glob, os, json, urllib.request

sys.path.insert(0, "python")
from minisgl.models.gguf_reader import GgufReader
from minisgl.tokenizer.gguf_vocab import GGUFBPETokenizer

model_dir = "/home/hhy/Models/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2-GGUF"
all_shards = glob.glob(f"{model_dir}/*.gguf")
shards = [
    f for f in all_shards
    if "dflash" not in os.path.basename(f).lower()
    and "mmproj" not in os.path.basename(f).lower()
]
main = shards[0]

with GgufReader(main) as r:
    meta = r.metadata

tok = GGUFBPETokenizer(meta)


def oracle_tokenize(text: str) -> list[int]:
    req = urllib.request.Request(
        "http://127.0.0.1:8080/tokenize",
        data=json.dumps({"content": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["tokens"]


def oracle_detokenize(ids: list[int]) -> str:
    req = urllib.request.Request(
        "http://127.0.0.1:8080/detokenize",
        data=json.dumps({"tokens": ids}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["content"]


def compare(text: str, label: str = "") -> bool:
    our = tok.encode(text)
    oracle = oracle_tokenize(text)
    if our == oracle:
        print(f"  MATCH ({len(our)} tokens): {label or text[:60]!r}")
        return True
    print(f"  MISMATCH: {label or text[:60]!r}")
    print(f"    Our len={len(our)}, Oracle len={len(oracle)}")
    for i in range(min(len(our), len(oracle))):
        if our[i] != oracle[i]:
            print(f"    First diff at {i}:")
            print(f"      Our:    {our[i]} = {tok._tokens[our[i]]!r} (type={tok._token_type[our[i]]})")
            print(f"      Oracle: {oracle[i]} = {tok._tokens[oracle[i]]!r} (type={tok._token_type[oracle[i]]})")
            start = max(0, i - 3)
            end = min(len(our), i + 4)
            print(f"      Our context:    {our[start:end]}")
            print(f"      Oracle context: {oracle[start:end]}")
            break
    return False


# --- Test 1: Chat template ---
print("=== Chat template ===")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello! How are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you!"},
    {"role": "user", "content": "What is the capital of France?"},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
compare(prompt, "chat template (4 messages)")

# --- Test 2: Plain text ---
print("\n=== Plain text ===")
test_texts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "你好，世界！",
    "12345",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "Multi-line\nwith newlines\tand tabs",
    "Special chars: !@#$%^&*()_+-=[]{}|;:'\",.<>/?\\",
    "Mixed 中文 and English text",
    "   leading and trailing spaces   ",
    "CONTRACTIONS: don't, it's, we're, they've, I'm, you'll, I'd",
    "Numbers: 1 2 3 4 5 123 456 789",
    "URL: https://example.com/path?query=value",
    "Email: test@example.com",
    "Code: if (x > 0) { return x; }",
    "Math: a + b = c, x^2 + y^2 = z^2",
    "Punctuation: ... !!! ??? ;;;",
]
all_match = True
for text in test_texts:
    if not compare(text):
        all_match = False

# --- Test 2b: Special-token partitioning edge cases ---
print("\n=== Special-token partitioning ===")
# Build special tokens via chr() to avoid heredoc mangling
im_start = chr(0x3c) + chr(0x7c) + "im_start" + chr(0x7c) + chr(0x3e)  # 12 chars
im_end = chr(0x3c) + chr(0x7c) + "im_end" + chr(0x7c) + chr(0x3e)      # 10 chars
think = chr(0x3c) + "think" + chr(0x3e)                                  # 7 chars
think_end = chr(0x3c) + "/think" + chr(0x3e)                             # 8 chars

special_tests = [
    (im_start, "im_start alone"),
    (im_end, "im_end alone"),
    (think, "think alone"),
    (think_end, "think_end alone"),
    (im_start + "Hello" + im_end, "im_start + Hello + im_end"),
    (im_start + "Hello" + " " + im_end, "im_start + 'Hello ' + im_end"),
    (think + "Let me think..." + think_end, "think + text + think_end"),
    (im_start + im_end, "im_start + im_end (no text)"),
    ("Hello " + im_start + " world " + im_end, "Hello + im_start + world + im_end"),
    (im_start + im_start, "im_start x2"),
    ("before" + think + "middle" + think_end + "after", "before+think+middle+think_end+after"),
]
all_special_match = True
for text, label in special_tests:
    if not compare(text, label):
        all_special_match = False

# --- Test 3: Decode ---
print("\n=== Decode ===")
decode_tests = [
    "Hello, world!",
    "你好，世界！",
    "The quick brown fox jumps over the lazy dog.",
    "12345",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "Multi-line\nwith newlines\tand tabs",
    "Mixed 中文 and English text",
    "   leading and trailing spaces   ",
    "CONTRACTIONS: don't, it's, we're, they've, I'm, you'll, I'd",
    "URL: https://example.com/path?query=value",
    "Code: if (x > 0) { return x; }",
    "Math: a + b = c, x^2 + y^2 = z^2",
    "Punctuation: ... !!! ??? ;;;",
    "Special chars: !@#$%^&*()_+-=[]{}|;:'\",.<>/?\\",
    "Numbers: 1 2 3 4 5 123 456 789",
    "Email: test@example.com",
]
all_decode_match = True
for text in decode_tests:
    ids = tok.encode(text)
    our_dec = tok.decode(ids)
    oracle_dec = oracle_detokenize(ids)
    if our_dec == oracle_dec:
        print(f"  MATCH: {text[:50]!r}")
    else:
        all_decode_match = False
        print(f"  MISMATCH: {text[:50]!r}")
        print(f"    Our:    {our_dec!r}")
        print(f"    Oracle: {oracle_dec!r}")

# --- Test 4: Round-trip ---
print("\n=== Round-trip ===")
all_rt_match = True
for text in test_texts:
    ids = tok.encode(text)
    dec = tok.decode(ids)
    if dec == text:
        print(f"  MATCH: {text[:50]!r}")
    else:
        all_rt_match = False
        print(f"  MISMATCH: {text[:50]!r}")
        print(f"    Original: {text!r}")
        print(f"    Decoded:  {dec!r}")

# --- Summary ---
print("\n=== Summary ===")
print(f"Plain text: {'ALL MATCH' if all_match else 'SOME MISMATCH'}")
print(f"Special:    {'ALL MATCH' if all_special_match else 'SOME MISMATCH'}")
print(f"Decode:     {'ALL MATCH' if all_decode_match else 'SOME MISMATCH'}")
print(f"Round-trip: {'ALL MATCH' if all_rt_match else 'SOME MISMATCH'}")
