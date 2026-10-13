#!/usr/bin/env bash
# Spec-decode diagnostic probe: measure (a) sustained decode tok/s on a tiny
# prompt (discriminates accept-collapse vs bypass vs working) and (b) TTFT on
# an ~8k prompt (populate overhead). No debug env vars -> clean timing.
# Launches the engine under setsid and stops the whole process group so no
# worker leaks hold the gloo rendezvous port.
set -u
cd "$(dirname "$0")"

TS=$(date +%Y%m%d-%H%M%S)
DBG="logs/spec-probe-${TS}.log"
BASE="http://127.0.0.1:8081"
CURL="curl -s --noproxy '*'"

cleanup() {
  if [ -n "${EPID:-}" ] && kill -0 "$EPID" 2>/dev/null; then
    kill -TERM -- -"$EPID" 2>/dev/null
    for i in $(seq 1 15); do kill -0 "$EPID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- -"$EPID" 2>/dev/null
  fi
  printf '{"kind": "gpu-release"}\n' > .supervisor-action.tmp && mv -f .supervisor-action.tmp .supervisor-action
  echo "[probe] gpu-release written"
}
trap cleanup EXIT

echo "[probe] === evacuating GPU (gpu-hold) ==="
printf '{"kind": "gpu-hold"}\n' > .supervisor-action.tmp && mv -f .supervisor-action.tmp .supervisor-action
for i in $(seq 1 60); do
  if [ -f .gpu-held ]; then echo "[probe] GPU held after ${i}x3s"; break; fi
  if ! curl -s --noproxy '*' -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "[probe] origin already down after ${i}x3s (evacuating)"; break
  fi
  sleep 3
done
sleep 5

echo "[probe] === starting engine on 8081 (clean, no debug env) ==="
setsid nohup .venv/bin/python -m minisgl --port 8081 > "$DBG" 2>&1 &
EPID=$!
echo "[probe] engine pgid $EPID; logging to $DBG"

echo "[probe] waiting for readiness (1-token completion)..."
READY=0
for i in $(seq 1 90); do
  R=$($CURL -m 20 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
    -d '{"model":"x","messages":[{"role":"user","content":"Reply with the single word: ping"}],"max_tokens":1,"temperature":0}')
  if echo "$R" | grep -q '"choices"'; then READY=1; echo "[probe] ready after ${i}x5s"; break; fi
  if ! kill -0 "$EPID" 2>/dev/null; then echo "[probe] ENGINE DIED during startup"; tail -40 "$DBG"; break; fi
  sleep 5
done
if [ "$READY" -ne 1 ]; then echo "[probe] readiness FAILED"; exit 1; fi

echo "[probe] === request A: tiny prompt + 400 tokens (decode rate) ==="
START=$(date +%s.%N)
$CURL -m 300 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"Count from 1 to 400, one number per line."}],"max_tokens":400,"temperature":0}' \
  > "logs/spec-probe-outA-${TS}.json" 2>&1
END=$(date +%s.%N)
echo "[probe] A wall: $(echo "$END $START" | awk '{printf "%.1f s", $1-$2}')"

echo "[probe] === request B: ~8k prompt + 50 tokens (TTFT) ==="
.venv/bin/python - "$TS" <<'EOF'
import json, sys
ts = sys.argv[1]
para = ("The engine measures sustained decode throughput under a fixed benchmark ruler. "
        "Every step of the speculative path runs a full forward pass followed by a chain of "
        "draft hops and a verify pass that accepts the longest matching prefix. ") * 12
prompt = para * 13  # ~31 KB ~ 8k tokens
json.dump({"model": "x",
           "messages": [{"role": "user", "content": prompt + " Now answer: what is 2+2?"}],
           "max_tokens": 50, "temperature": 0},
          open(f"logs/spec-probe-promptB-{ts}.json", "w"))
EOF
START=$(date +%s.%N)
$CURL -m 300 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d @"logs/spec-probe-promptB-${TS}.json" > "logs/spec-probe-outB-${TS}.json" 2>&1
END=$(date +%s.%N)
echo "[probe] B wall: $(echo "$END $START" | awk '{printf "%.1f s", $1-$2}')"

echo "[probe] === engine request lines ==="
grep -a "elapsed=" "$DBG" | grep -a "completions" | tail -6 | cut -c1-200
echo "[probe] === errors? ==="
grep -aiE "error|traceback|assert|cuda" "$DBG" | head -15 | cut -c1-200
echo "[probe] === stray check (must be empty) ==="
ss -tlnp 2>/dev/null | grep ":8082" || echo "(8082 free)"

echo "[probe] === stopping engine (process group) ==="
kill -TERM -- -"$EPID" 2>/dev/null
for i in $(seq 1 15); do kill -0 "$EPID" 2>/dev/null || break; sleep 1; done
kill -KILL -- -"$EPID" 2>/dev/null
sleep 3
ss -tlnp 2>/dev/null | grep ":8082" && echo "[probe] WARNING: 8082 still held" || echo "[probe] 8082 free after stop"
echo "[probe] DONE (log: $DBG)"
