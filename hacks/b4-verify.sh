#!/usr/bin/env bash
# B4 correctness gate (supervisor gpu-job command): launch the engine on a
# spare port, wait for readiness, then run a short completion on a prompt with
# a known answer and check the output is coherent (not garbage). Self-terminating.
#
# This is the end-to-end check for bug B (the GGUF reader's tensor-data base
# was 11 bytes too small, so every weight was misread). With the fix, a
# factual prompt should produce a coherent answer, not token soup.
set -u
cd "$(dirname "$0")/.."

PORT=8124
LOG=logs/b4-verify.log
mkdir -p logs
: > "$LOG"

.venv/bin/python -m minisgl --port "$PORT" >> "$LOG" 2>&1 &
PID=$!
echo "b4-verify: engine launched (pid $PID) on port $PORT, log $LOG"

# Wait for readiness (a 1-token completion), up to 30 min.
ready=0
deadline=$(( $(date +%s) + 1800 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "b4-verify: engine process exited before serving"
    break
  fi
  if curl -s -m 120 -o /tmp/b4-ready.json \
       -H 'content-type: application/json' \
       -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
       "http://127.0.0.1:$PORT/v1/chat/completions" 2>/dev/null \
     && grep -q '"choices"' /tmp/b4-ready.json 2>/dev/null; then
    ready=1
    break
  fi
  sleep 2
done

if [ "$ready" -ne 1 ]; then
  echo "b4-verify: FAIL (engine not ready; see $LOG)"
  kill -TERM -- "-$PID" 2>/dev/null
  exit 1
fi
echo "b4-verify: engine ready"

# Short completion on a factual prompt (32 tokens, default sampling).
curl -s -m 300 -o /tmp/b4-verify.json \
     -H 'content-type: application/json' \
     -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":32}' \
     "http://127.0.0.1:$PORT/v1/chat/completions" 2>/dev/null

# Kill the engine tree (setsid: negative PID = the whole process group).
kill -TERM -- "-$PID" 2>/dev/null
for _ in $(seq 1 30); do
  kill -0 "$PID" 2>/dev/null || break
  sleep 1
done
kill -KILL -- "-$PID" 2>/dev/null

echo "b4-verify: response:"
cat /tmp/b4-verify.json 2>/dev/null
echo ""

# Coherence check: the answer should contain "Paris" (case-insensitive).
if grep -qi 'paris' /tmp/b4-verify.json 2>/dev/null; then
  echo "b4-verify: PASS (coherent answer contains 'Paris')"
  exit 0
else
  echo "b4-verify: FAIL (answer does not contain 'Paris'; output may be garbage)"
  exit 1
fi
