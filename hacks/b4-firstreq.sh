#!/usr/bin/env bash
# Diagnostic (scratch): launch the engine, poll the LOG for "Scheduler is ready"
# (no model request), then run the 32-token completion TWICE as the first and
# second model requests. Reports both outputs so we can tell whether the bug is
# in the forward pass (both garbage) or between-request state (req1 ok, req2 bad).
set -u
cd "$(dirname "$0")/.."

PORT=8125
LOG=logs/b4-firstreq.log
mkdir -p logs
: > "$LOG"

.venv/bin/python -m minisgl --port "$PORT" >> "$LOG" 2>&1 &
PID=$!
echo "b4-firstreq: engine launched (pid $PID) on port $PORT, log $LOG"

# Wait for the scheduler to report ready in the LOG (no model inference).
ready=0
deadline=$(( $(date +%s) + 1800 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "b4-firstreq: engine process exited before serving"
    break
  fi
  if grep -q "Scheduler is ready" "$LOG" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 2
done

if [ "$ready" -ne 1 ]; then
  echo "b4-firstreq: FAIL (engine not ready; see $LOG)"
  kill -TERM -- "-$PID" 2>/dev/null
  exit 1
fi
echo "b4-firstreq: engine ready (log marker)"

# Request 1: fresh state (first model request ever).
curl -s -m 300 -o /tmp/b4r1.json \
     -H 'content-type: application/json' \
     -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":32}' \
     "http://127.0.0.1:$PORT/v1/chat/completions" 2>/dev/null
# Request 2: contaminated state (second model request).
curl -s -m 300 -o /tmp/b4r2.json \
     -H 'content-type: application/json' \
     -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":32}' \
     "http://127.0.0.1:$PORT/v1/chat/completions" 2>/dev/null

kill -TERM -- "-$PID" 2>/dev/null
for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
kill -KILL -- "-$PID" 2>/dev/null

echo "=== REQUEST 1 (fresh state) ==="
cat /tmp/b4r1.json 2>/dev/null
echo ""
echo "=== REQUEST 2 (contaminated state) ==="
cat /tmp/b4r2.json 2>/dev/null
echo ""
if grep -qi 'paris' /tmp/b4r1.json 2>/dev/null; then echo "REQ1: contains Paris"; else echo "REQ1: NO Paris"; fi
if grep -qi 'paris' /tmp/b4r2.json 2>/dev/null; then echo "REQ2: contains Paris"; else echo "REQ2: NO Paris"; fi
exit 0
