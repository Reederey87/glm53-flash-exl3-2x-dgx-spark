#!/usr/bin/env bash
# DFlash2 housekeep cluster smoke: runtime identity + KV + error triage.
set -uo pipefail
D=/home/nvidia/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
cd "$D" || exit 1

echo "=== A. acceptance.sh ==="
bash local/acceptance.sh 2>&1 | tail -22

echo
echo "=== B. runtime identity in the head container ==="
docker exec glm53-exl3-head python3 /opt/glm53/smoke_identity.py 2>&1 | grep -vE "^INFO|^WARNING|Triton is installed|Triton not installed"

echo
echo "=== C. JIT monitor / first-launch Triton JIT ==="
docker logs glm53-exl3-head 2>&1 | grep -E "jit_monitor|JIT" | tail -6

echo
echo "=== D. KV pool (head) ==="
curl -s http://127.0.0.1:8000/metrics 2>/dev/null \
  | grep -oE 'kv_cache_size_tokens="[0-9]+"|kv_cache_max_concurrency="[0-9.]+"|num_gpu_blocks="[0-9]+"|kv_cache_memory_bytes="[0-9]+"' | sort -u
echo "--- head log KV lines ---"
docker logs glm53-exl3-head 2>&1 | grep -E "GPU KV cache size|DFlash2 drafter KV" | tail -3

echo
echo "=== E. MemFree tripwire ==="
echo "head:   $(awk '/MemFree/{print $2}' /proc/meminfo) kB"
ssh -o BatchMode=yes -o ConnectTimeout=10 nvidia@192.168.177.11 \
  "echo worker: \$(awk '/MemFree/{print \$2}' /proc/meminfo) kB"

echo
echo "=== F. error triage ==="
echo "head tracebacks/errors:   $(docker logs glm53-exl3-head 2>&1 | grep -ciE 'traceback|cuda error|illegal memory|AssertionError')"
echo "worker tracebacks/errors: $(ssh -o BatchMode=yes -o ConnectTimeout=10 nvidia@192.168.177.11 "docker logs glm53-exl3-worker 2>&1 | grep -ciE 'traceback|cuda error|illegal memory|AssertionError'")"
echo "--- head error lines (first 10) ---"
docker logs glm53-exl3-head 2>&1 | grep -iE 'traceback|cuda error|illegal memory|AssertionError' | head -10
echo "--- worker error lines (first 10) ---"
ssh -o BatchMode=yes -o ConnectTimeout=10 nvidia@192.168.177.11 \
  "docker logs glm53-exl3-worker 2>&1 | grep -iE 'traceback|cuda error|illegal memory|AssertionError' | head -10"

echo
echo "=== G. serving identity ==="
docker inspect -f 'head image={{.Config.Image}} id={{.Image}}' glm53-exl3-head
ssh -o BatchMode=yes -o ConnectTimeout=10 nvidia@192.168.177.11 \
  "docker inspect -f 'worker image={{.Config.Image}} id={{.Image}}' glm53-exl3-worker"
echo "=== SMOKE DONE ==="
