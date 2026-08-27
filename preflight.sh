#!/usr/bin/env bash
# Pre-start guards, run ON each node by systemd. Usage: preflight.sh <head|worker>
set -euo pipefail
ROLE="${1:?usage: preflight.sh <head|worker>}"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/cluster.env"

fail() { echo "preflight FAIL: $*" >&2; exit 1; }
wait_for() { # wait_for <secs> <label> <cmd...>
  local end=$(( $(date +%s) + $1 )) label="$2"; shift 2
  until "$@" >/dev/null 2>&1; do
    [ "$(date +%s)" -ge "$end" ] && fail "timed out waiting for $label"
    sleep 5
  done
  echo "preflight ok: $label"
}

case "$ROLE" in
  head)   MY_IP="$HEAD_R1";   PEER_IP="$WORKER_R1" ;;
  worker) MY_IP="$WORKER_R1"; PEER_IP="$HEAD_R1" ;;
  *) fail "unknown role $ROLE" ;;
esac

wait_for 300 "QSFP address $MY_IP on $QSFP_IF" \
  sh -c "ip -4 -o addr show dev $QSFP_IF | grep -Fq '$MY_IP/'"
wait_for 300 "docker daemon" docker info
[ -e /dev/infiniband ] || fail "/dev/infiniband missing (RDMA not available)"

docker image inspect "$GLM53_IMAGE" >/dev/null 2>&1 \
  || fail "image $GLM53_IMAGE not present -- run image/build.sh on this node"

MODEL_DIR="$HF_CACHE/hub/models--${GLM53_MODEL//\//--}"
if ! find "$MODEL_DIR" -name config.json -print -quit 2>/dev/null | grep -q .; then
  fail "weights not found under $MODEL_DIR -- see docs/04-bringup.md"
fi

# vm.min_free_kbytes reserves memory away from the GPU on this UMA platform, so
# the two nodes MUST agree or one will size its KV cache differently.
MINFREE="$(cat /proc/sys/vm/min_free_kbytes)"
if [ "$ROLE" = head ]; then
  # docs/01-architecture.md calls this invariant mandatory, so enforce it rather
  # than print it. A mismatch makes the two ranks size their KV caches
  # differently and surfaces as a phantom memory failure AFTER a 12-minute load,
  # with nothing in the logs pointing at the sysctl.
  PEERFREE="$(ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
                "$CLUSTER_USER@$PEER_IP" 'cat /proc/sys/vm/min_free_kbytes' 2>/dev/null || echo unknown)"
  if [ "$PEERFREE" = unknown ]; then
    echo "preflight WARN: could not read peer vm.min_free_kbytes" >&2
  elif [ "$MINFREE" != "$PEERFREE" ]; then
    fail "vm.min_free_kbytes differs: this node $MINFREE, peer $PEERFREE -- set both the same"
  else
    echo "preflight ok: vm.min_free_kbytes=$MINFREE matches the peer"
  fi
else
  echo "preflight ok: vm.min_free_kbytes=$MINFREE"
fi

# Refuse to run by hand against a live service: this removes the container, and
# with Restart=on-failure that triggers a ~12 min reload from what the operator
# thought was a read-only check. INVOCATION_ID is set only under systemd.
if [ -z "${INVOCATION_ID:-}" ] && docker ps -q -f name=^vllm-glm53$ | grep -q .; then
  fail "vllm-glm53 is RUNNING and this was not started by systemd. Refusing to remove it."
fi
docker rm -f vllm-glm53 >/dev/null 2>&1 || true

if [ "$ROLE" = head ]; then
  wait_for 300 "peer $PEER_IP reachable over QSFP" \
    ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
        "$CLUSTER_USER@$PEER_IP" true
  wait_for 600 "worker container up" \
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$CLUSTER_USER@$PEER_IP" \
        'docker ps -q -f name=^vllm-glm53$ | grep -q .'
fi
echo "preflight ok: $ROLE"
