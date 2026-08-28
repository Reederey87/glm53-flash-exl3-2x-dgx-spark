#!/usr/bin/env bash
# Correctness gate. Runs smoke.py on the head against the loopback API.
#   bash bench/smoke.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/cluster.env"; source "$ROOT/lib.sh"
node_ssh head "PYTHONUNBUFFERED=1 python3 - --url http://127.0.0.1:$API_PORT --model $SERVED_MODEL_NAME --max-model-len $MAX_MODEL_LEN" \
  < "$ROOT/bench/smoke.py"
