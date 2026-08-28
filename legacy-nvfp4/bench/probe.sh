#!/usr/bin/env bash
# Throughput probe. Warm the engine first (run bench/smoke.sh once) -- the first
# generations after any boot pay JIT compilation and are not a baseline.
#   bash bench/probe.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/cluster.env"; source "$ROOT/lib.sh"
mkdir -p "$ROOT/results"
OUT="$ROOT/results/probe-$(date -u +%Y%m%dT%H%M%SZ).jsonl"

run() { # run <concurrency> <reps> <label>
  node_ssh head "python3 - --url http://127.0.0.1:$API_PORT --model $SERVED_MODEL_NAME \
    --concurrency $1 --reps $2 --max-tokens 2048 --label $3" < "$ROOT/bench/cbench.py" | tee -a "$OUT"
}
echo "== single stream x3"; run 1 3 c1
echo "== 8 concurrent x3 (queues behind max-num-seqs)"; run 8 3 c8
echo "== results: $OUT"
