#!/usr/bin/env bash
# Stop the pair, head first. Disarms the watchdog BEFORE touching anything --
# a deliberate teardown looks exactly like a wedged engine to it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/cluster.env"; source "$ROOT/lib.sh"

# Stop the SERVICE as well as the timer -- stopping a timer does not interrupt a
# bounce already in progress, which would restart the pair moments after we
# report it stopped.
node_ssh head 'systemctl --user stop vllm-glm53-watchdog.timer vllm-glm53-watchdog.service' >/dev/null 2>&1 || true
echo "== watchdog disarmed"
node_ssh head 'systemctl --user stop vllm-glm53-head.service'   || echo "WARN: head stop non-zero" >&2
node_ssh worker 'systemctl --user stop vllm-glm53-worker.service' || echo "WARN: worker stop non-zero" >&2
for role in head worker; do
  node_ssh "$role" 'docker rm -f vllm-glm53 >/dev/null 2>&1 || true'
done
echo "== stopped"
