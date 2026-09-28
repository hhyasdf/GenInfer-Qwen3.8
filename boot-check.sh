#!/usr/bin/env bash
# B3 boot gate (supervisor gpu-job command): launch the engine on a spare
# port, wait for a 1-token chat completion, then kill the engine tree.
# Self-terminating — the supervisor polls until this script exits.
#
# The engine is setsid'd by minisgl.__main__ (PGID == PID), so the whole
# process tree (2 scheduler ranks + detokenizer) dies with `kill -TERM -$PID`.
set -u
cd "$(dirname "$0")"

PORT=8123
LOG=logs/boot-check.log
mkdir -p logs
: > "$LOG"

.venv/bin/python -m minisgl --port "$PORT" >> "$LOG" 2>&1 &
PID=$!
echo "boot-check: engine launched (pid $PID) on port $PORT, log $LOG"

ok=0
deadline=$(( $(date +%s) + 1800 ))  # 30 min: 23.6 GB GGUF load + first JIT compiles
while [ "$(date +%s)" -lt "$deadline" ]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "boot-check: engine process exited before serving"
    break
  fi
  if curl -s -m 120 -o /tmp/boot-check-1tok.json \
       -H 'content-type: application/json' \
       -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
       "http://127.0.0.1:$PORT/v1/chat/completions" 2>/dev/null \
     && grep -q '"choices"' /tmp/boot-check-1tok.json 2>/dev/null; then
    ok=1
    break
  fi
  sleep 2
done

# Kill the engine tree (setsid: negative PID = the whole process group).
kill -TERM -- "-$PID" 2>/dev/null
for _ in $(seq 1 30); do
  kill -0 "$PID" 2>/dev/null || break
  sleep 1
done
kill -KILL -- "-$PID" 2>/dev/null

if [ "$ok" -eq 1 ]; then
  echo "boot-check: PASS (1-token completion on port $PORT)"
  cat /tmp/boot-check-1tok.json
  exit 0
else
  echo "boot-check: FAIL (no 1-token completion on port $PORT; see $LOG)"
  exit 1
fi
