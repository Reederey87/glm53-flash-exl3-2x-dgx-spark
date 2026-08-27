#!/usr/bin/env bash
# Build the TP=2 runtime image on BOTH nodes (each must build natively: the AOT
# module is compiled for sm_121a). Expect tens of minutes.
#
#   bash image/build.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/cluster.env"
# shellcheck disable=SC1091
source "$ROOT/lib.sh"

TAG="${GLM53_IMAGE:-glm53-flash-nvfp4-tp2:local}"
CTX="$KIT_DIR/image"

for role in head worker; do
  if [ -n "$(node_ssh "$role" 'docker ps -q -f name=^vllm-glm53$' 2>/dev/null)" ]; then
    echo "FAIL: vllm-glm53 is running on $(node_label "$role") -- the build needs the RAM" >&2
    exit 1
  fi
done

echo "== syncing build context"
for role in head worker; do
  node_rsync "$role" "$ROOT/image/" "$CTX/"
done

declare -a PIDS=()
for role in head worker; do
  echo "== building $TAG on $(node_label "$role")"
  ( node_ssh "$role" "cd '$CTX' && docker build --network=host -t '$TAG' ." \
      > "$ROOT/build-$role.log" 2>&1 ) &
  PIDS+=("$!")
done

rc=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}" || { echo "BUILD FAILED on node #$i -- see build-*.log" >&2; rc=1; }
done
[ "$rc" -eq 0 ] || exit 1

echo "== verifying the specialization landed on both nodes"
for role in head worker; do
  node_ssh "$role" "docker run --rm --entrypoint bash '$TAG' -lc \
    'GLM53_GB10_PATCH_DISABLE=1 python3 /opt/glm53-tp2/verify_tp2.py'"
done
echo "ok: $TAG built on both nodes"
