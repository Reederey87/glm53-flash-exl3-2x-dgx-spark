#!/usr/bin/env bash
# DFlash2 housekeep cluster smoke: runtime identity + KV + error triage.
#
# Exit status is the contract: 0 only when acceptance AND the identity probe
# pass. Diagnostic greps that may legitimately match nothing (JIT monitor lines,
# error triage) are explicitly nonfatal -- they report facts, they are not gates.
#
# Overridable for the host regression test in tests/test_dflash2_smoke_harness.py:
#   GLM53_KIT_DIR   kit root to cd into (default: the Spark kit path)
#   SMOKE_PROBE     identity probe to stream into the container
#   HEAD_CONTAINER  head container name
set -uo pipefail

D="${GLM53_KIT_DIR:-/home/nvidia/GLM-5.3-Flash-EXL3-2x-DGX-Sparks}"
cd "$D" || exit 1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="${SMOKE_PROBE:-$HERE/smoke_identity.py}"
HEAD_CONTAINER="${HEAD_CONTAINER:-glm53-exl3-head}"
WORKER_SSH="${WORKER_SSH:-nvidia@192.168.177.11}"
WORKER_CONTAINER="${WORKER_CONTAINER:-glm53-exl3-worker}"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

rc=0

echo "=== A. acceptance.sh ==="
if bash local/acceptance.sh > "$TMP/acceptance.log" 2>&1; then
    tail -22 "$TMP/acceptance.log"
else
    tail -22 "$TMP/acceptance.log"
    echo "FAILED: acceptance.sh exited non-zero"
    rc=1
fi

echo
echo "=== B. runtime identity in the head container ==="
# The probe is streamed in over stdin: nothing in the image copies it, so
# referencing /opt/glm53/smoke_identity.py would fail on a fresh container.
if docker exec -i "$HEAD_CONTAINER" python3 - < "$PROBE" > "$TMP/identity.log" 2>&1; then
    ident_rc=0
else
    ident_rc=$?
fi
grep -vE "^INFO|^WARNING|Triton is installed|Triton not installed" "$TMP/identity.log" || true
if [ "$ident_rc" -ne 0 ]; then
    echo "FAILED: identity probe exited $ident_rc"
    rc=1
fi

echo
echo "=== C. JIT monitor / first-launch Triton JIT ==="
docker logs "$HEAD_CONTAINER" 2>&1 | grep -E "jit_monitor|JIT" | tail -6 || true

echo
echo "=== D. KV pool (head) ==="
curl -s http://127.0.0.1:8000/metrics 2>/dev/null \
  | grep -oE 'kv_cache_size_tokens="[0-9]+"|kv_cache_max_concurrency="[0-9.]+"|num_gpu_blocks="[0-9]+"|kv_cache_memory_bytes="[0-9]+"' | sort -u || true
echo "--- head log KV lines ---"
docker logs "$HEAD_CONTAINER" 2>&1 | grep -E "GPU KV cache size|DFlash2 drafter KV" | tail -3 || true

echo
echo "=== E. MemFree tripwire ==="
echo "head:   $(awk '/MemFree/{print $2}' /proc/meminfo) kB"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" \
  "echo worker: \$(awk '/MemFree/{print \$2}' /proc/meminfo) kB" || { echo "FAILED: worker MemFree probe"; rc=1; }

echo
echo "=== F. error triage ==="
# Facts, not gates: an empty result here is the good outcome.
echo "head tracebacks/errors:   $(docker logs "$HEAD_CONTAINER" 2>&1 | grep -ciE 'traceback|cuda error|illegal memory|AssertionError')"
echo "worker tracebacks/errors: $(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "docker logs $WORKER_CONTAINER 2>&1 | grep -ciE 'traceback|cuda error|illegal memory|AssertionError'")"
echo "--- head error lines (first 10) ---"
docker logs "$HEAD_CONTAINER" 2>&1 | grep -iE 'traceback|cuda error|illegal memory|AssertionError' | head -10 || true
echo "--- worker error lines (first 10) ---"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" \
  "docker logs $WORKER_CONTAINER 2>&1 | grep -iE 'traceback|cuda error|illegal memory|AssertionError' | head -10" || true

echo
echo "=== G. serving identity ==="
docker inspect -f 'head image={{.Config.Image}} id={{.Image}}' "$HEAD_CONTAINER" || rc=1
ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" \
  "docker inspect -f 'worker image={{.Config.Image}} id={{.Image}}' $WORKER_CONTAINER" || rc=1

if [ "$rc" -eq 0 ]; then
    echo "=== SMOKE PASS ==="
else
    echo "=== SMOKE FAIL (required check failed above) ==="
fi
exit "$rc"
