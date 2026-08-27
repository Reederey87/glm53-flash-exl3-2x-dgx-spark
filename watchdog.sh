#!/usr/bin/env bash
# Inference-level watchdog for the head.
#
# WHY: the API process survives a lost tensor-parallel peer, so /health keeps
# answering 200 while inference is wedged. Only a real completion proves the
# engine works. If /health is up but a 1-token request times out, bounce the
# PAIR in strict order -- head stop, worker restart, head start -- because the
# head's stale rendezvous store must be gone before the worker rejoins, or the
# fresh worker joins a dead group and never exits.
set -uo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/cluster.env"

PROBE_TIMEOUT="${PROBE_TIMEOUT:-90}"
URL="http://127.0.0.1:$API_PORT"

# Still loading? Not our problem -- /health is down during startup.
curl -fsS --max-time 5 "$URL/health" >/dev/null 2>&1 || exit 0

body="$(python3 -c 'import json,sys; print(json.dumps({
  "model": sys.argv[1], "messages": [{"role":"user","content":"hi"}],
  "max_tokens": 1, "temperature": 1}))' "$SERVED_MODEL_NAME")"

if curl -fsS --max-time "$PROBE_TIMEOUT" -H 'Content-Type: application/json' \
     -d "$body" "$URL/v1/chat/completions" >/dev/null 2>&1; then
  exit 0
fi

echo "watchdog: /health OK but inference timed out -- bouncing the pair" >&2
systemctl --user stop vllm-glm53-head.service
for _ in $(seq 1 18); do docker inspect vllm-glm53 >/dev/null 2>&1 || break; sleep 5; done
docker rm -f vllm-glm53 >/dev/null 2>&1 || true
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
  "$CLUSTER_USER@$WORKER_R1" \
  'systemctl --user reset-failed vllm-glm53-worker.service 2>/dev/null; systemctl --user restart vllm-glm53-worker.service' \
  || echo "watchdog: worker restart failed; starting head anyway (preflight will wait)" >&2
systemctl --user reset-failed vllm-glm53-head.service 2>/dev/null || true
systemctl --user start vllm-glm53-head.service
