#!/usr/bin/env bash
# Bring the pair up, in order, from your Mac (or any machine with ssh to both).
#   ./start-cluster.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/cluster.env"; source "$ROOT/lib.sh"
HEALTH_WAIT="${HEALTH_WAIT_SECS:-3600}"

# A failed boot leaves the WORKER unit active and headless -- it does not exit
# when the head dies, and `systemctl start` on an active unit is a silent no-op.
# Without this teardown a retry rendezvouses with a stale worker still holding
# ~90 GiB and hangs at parallel_state init. Always start from a known state.
echo "== clearing any previous state on both nodes"
for role in head worker; do
  node_ssh "$role" "systemctl --user stop vllm-glm53-$role.service" >/dev/null 2>&1 || true
  node_ssh "$role" 'docker rm -f vllm-glm53 >/dev/null 2>&1 || true'
done

echo "== starting worker (headless; it waits for the head to appear)"
node_ssh worker 'systemctl --user start vllm-glm53-worker.service'
echo "== starting head (its preflight waits for the worker)"
node_ssh head 'systemctl --user start vllm-glm53-head.service'

echo "== waiting for /health on ${API_PORT} (weights are ~181 GiB; allow ~12 min)"
deadline=$(( $(date +%s) + HEALTH_WAIT ))
while :; do
  if node_ssh head "curl -fsS --max-time 5 http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
    echo "== serving:"
    node_ssh head "curl -fsS http://127.0.0.1:$API_PORT/v1/models"; echo
    node_ssh head 'systemctl --user start vllm-glm53-watchdog.timer' 2>/dev/null \
      && echo "== watchdog armed" || true
    exit 0
  fi
  for role in head worker; do
    if [ "$(node_ssh "$role" "systemctl --user is-active vllm-glm53-$role.service" 2>/dev/null)" = failed ]; then
      echo "ERROR: vllm-glm53-$role failed. Recent log:" >&2
      node_ssh "$role" "journalctl --user -u vllm-glm53-$role.service -n 40 --no-pager" >&2 || true
      exit 1
    fi
  done
  [ "$(date +%s)" -ge "$deadline" ] && { echo "ERROR: timed out waiting for /health" >&2; exit 1; }
  sleep 15
done
