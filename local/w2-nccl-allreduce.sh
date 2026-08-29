#!/usr/bin/env bash
# LOCAL to this cluster; not part of the vendored upstream kit.
#
# W2 — NCCL all_reduce measurement (radar plan, docs/06-improvement-plan.md).
# Run ON spark1 as nvidia, with production STOPPED on both nodes:
#
#   systemctl --user stop vllm-glm53exl3-watchdog.timer
#   systemctl --user stop vllm-glm53exl3.service
#   bash ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/local/w2-nccl-allreduce.sh
#
# Measures NCCL-layer all_reduce bus bandwidth over the CX7 fabric, twice:
#   leg A: single-rail (prod config: rocep1s0f1, MERGE_NICS=0)
#   leg B: dual-rail   (rocep1s0f1 + roceP2p1s0f1, MERGE_NICS=1)
# Env mirrors start.sh's nccl_common exactly (RoCEv2, GID 3, NVLS/CUMEM off).
# Sweep 8B..256MB covers everything TP=2 moves here: decode all_reduces are
# tens-of-KB (latency-bound); a full 3584-token prefill chunk is tens-of-MB.
#
# REFUSES to run if a glm53 container is up on either node — the test wants
# the GPUs and would fight the serving pair for unified memory.
set -euo pipefail

WORKER=192.168.177.11
IF=enp1s0f1np1
BIN=/home/nvidia/nccl-tests/build/all_reduce_perf
LIBS=/home/nvidia/nccl/build/lib:/usr/local/cuda/lib64
OUT="$HOME/w2-nccl-allreduce-$(date -u +%Y%m%dT%H%M%SZ).log"

die() { echo "FATAL: $*" >&2; exit 1; }

[ -x "$BIN" ] || die "missing $BIN"
ssh -o BatchMode=yes "nvidia@$WORKER" "test -x $BIN" || die "missing $BIN on worker"
docker ps --format '{{.Names}}' | grep -q glm53 && die "glm53 container running on head — stop production first"
ssh -o BatchMode=yes "nvidia@$WORKER" "docker ps --format '{{.Names}}' | grep -q glm53" \
    && die "glm53 container running on worker — stop production first"

run_leg() { # $1=label $2=NCCL_IB_HCA $3=NCCL_IB_MERGE_NICS
    echo "=== leg $1  (HCA=$2 MERGE_NICS=$3) ===" | tee -a "$OUT"
    mpirun -np 2 -H "192.168.177.10:1,${WORKER}:1" \
        --mca btl_tcp_if_include "$IF" --mca oob_tcp_if_include "$IF" \
        -x "LD_LIBRARY_PATH=$LIBS" \
        -x NCCL_IB_DISABLE=0 -x NCCL_IB_ROCE_VERSION_NUM=2 -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_NET=IB -x NCCL_NET_PLUGIN=none \
        -x NCCL_NVLS_ENABLE=0 -x NCCL_CUMEM_ENABLE=0 \
        -x NCCL_IGNORE_CPU_AFFINITY=1 -x NCCL_CROSS_NIC=0 \
        -x "NCCL_IB_HCA=$2" -x "NCCL_IB_MERGE_NICS=$3" \
        -x "NCCL_SOCKET_IFNAME=$IF" -x NCCL_DEBUG=WARN \
        "$BIN" -b 8 -e 268435456 -f 2 -g 1 -w 5 -n 20 2>&1 | tee -a "$OUT"
    echo | tee -a "$OUT"
}

echo "W2 NCCL all_reduce sweep  $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$OUT"
run_leg A-single-rail "rocep1s0f1" 0
run_leg B-dual-rail   "rocep1s0f1,roceP2p1s0f1" 1
echo "Log: $OUT"
