#!/usr/bin/env bash
# prod-start.sh — production entrypoint: stop, WAIT FOR MEMORY TO SETTLE, start.
#
# LOCAL to this cluster; not part of the upstream MiaAI-Lab kit.
#
# WHY THIS EXISTS (measured 2026-08-28): `start.sh restart` tears the pair down
# and starts the new one immediately. The kernel has not yet returned the old
# instance's unified memory, so vLLM's startup pre-check fails:
#
#   ValueError: Free memory on device cuda:0 (99.33/121.69 GiB) on startup is
#   less than desired GPU memory utilization (0.87, 105.87 GiB).
#
# That is vLLM's own gate, NOT an OOM (there is no OOM killer on these nodes).
# Sixty seconds later the same node reported 111 GiB available. So the fix is to
# WAIT for the memory to come back rather than to lower the gate — lowering it
# past true need would only convert a clean pre-check failure into a later OOM.
#
# The wait cannot live in ExecStartPre: that runs BEFORE ExecStart, i.e. before
# the stop that frees the memory. It has to sit between stop and start.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

WORKER_SSH="${WORKER_SSH:-nvidia@192.168.177.11}"
# Gate = GPU_MEM_UTIL x total. Wait for a little more than vLLM will demand.
# NOTE ON WHAT THIS GATE CAN AND CANNOT SEE:
# vLLM gates on CUDA device-free (torch.cuda.mem_get_info), which on this
# unified-memory box EXCLUDES reclaimable page cache. MemFree tracks it
# closely (measured: MemFree 96.47 vs cuda_free 95.94); MemAvailable does NOT
# (111.91 at the same instant). So gate on MemFree -- an earlier version used
# MemAvailable and happily green-lit starts that vLLM then refused.
# This wait exists for the TEARDOWN TRANSIENT (82 GiB not yet returned right
# after a stop). It cannot conjure memory that is held as page cache; the
# boot gate itself is sized for that via GPU_MEM_UTIL in .env.
# CALIBRATION: this guard exists to catch the TEARDOWN TRANSIENT -- 82 GiB of
# weights not yet returned by the kernel, which shows up as MemFree in the
# teens. It is NOT meant to gate on the last few GiB: measured steady-state
# MemFree with both nodes idle is 93-97 GiB (the rest is page cache that is
# never reclaimed at idle), so a threshold of 100 would block forever.
NEED_GIB="${NEED_GIB:-90}"
SETTLE_TIMEOUT="${SETTLE_TIMEOUT:-600}"
SETTLE_INTERVAL="${SETTLE_INTERVAL:-10}"

log() { echo "[prod-start] $*"; }

log "stopping any running pair (idempotent)"
./start.sh stop || true

avail_gib() { # MemFree: the closest proxy to what CUDA reports as device-free.
    awk '/^MemFree:/ {printf "%d", $2/1048576}' /proc/meminfo
}
avail_gib_worker() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" \
        "awk '/^MemFree:/ {printf \"%d\", \$2/1048576}' /proc/meminfo" 2>/dev/null
}

log "waiting for >= ${NEED_GIB} GiB MemFree on BOTH nodes (timeout ${SETTLE_TIMEOUT}s)"
deadline=$(( $(date +%s) + SETTLE_TIMEOUT ))
while :; do
    h="$(avail_gib)"; w="$(avail_gib_worker)"
    if [[ "$h" =~ ^[0-9]+$ ]] && [[ "$w" =~ ^[0-9]+$ ]] \
       && [ "$h" -ge "$NEED_GIB" ] && [ "$w" -ge "$NEED_GIB" ]; then
        log "memory settled: head ${h} GiB, worker ${w} GiB — starting"
        break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        # Proceed anyway: vLLM's own pre-check is the real gate and will fail
        # cleanly with a precise number. Better that than silently never starting.
        log "WARN: timed out waiting to settle (head=${h:-?} worker=${w:-?} GiB, need ${NEED_GIB})"
        log "WARN: starting anyway — vLLM's pre-check will report the exact shortfall"
        break
    fi
    sleep "$SETTLE_INTERVAL"
done

log "starting pair"
# --- JIT-cache config-shape guard (added 2026-08-28) -------------------------
# The persistent Triton/TileLang caches (upstream a099743) are safe across
# identical-config boots but MEASURED UNSAFE across spec-config changes:
# after a DFLASH_TOKENS 7->5->7 A/B, structured acceptance collapsed 0.96->0.58
# and recovered only after wiping both caches on both nodes (upstream #41871
# class). Hash the shape-affecting knobs; on change, wipe triton+tilelang on
# BOTH nodes (via the image, as root — the container writes them as root).
# DFLASH_MODEL/DFLASH_REVISION are in the hash too: a drafter checkpoint swap
# changes the drafter kernel shapes — same wipe class as DFLASH_TOKENS.
# The launcher's own default pin line is hashed too, so a default change in
# start.sh still wipes even when .env carries no DFLASH_REVISION= line.
shape_hash="$( { grep -E '^(DFLASH_TOKENS|DFLASH_DRAFT_TP|DFLASH_MODEL|DFLASH_REVISION|MTP_TOKENS|SPEC_METHOD|MAX_NUM_BATCHED_TOKENS|MAX_NUM_SEQS|MAX_MODEL_LEN|IMAGE|EXTRA_ARGS)=' .env 2>/dev/null; grep -E '^DFLASH_REVISION=' start.sh 2>/dev/null; } | sort | md5sum | cut -d' ' -f1)"
stamp="$HOME/.cache/vllm-glm53-flash/.config-shape"
if [ -n "$shape_hash" ] && [ "$(cat "$stamp" 2>/dev/null)" != "$shape_hash" ]; then
    echo "[prod-start] config shape changed — wiping Triton/TileLang JIT caches on both nodes"
    # Resolve the wipe container from the IMAGE= line verbatim (the self-built
    # image has no ghcr digest — the old digest-only grep matched nothing and
    # the wipe silently no-op'd while the stamp still advanced). Fallback: any
    # ghcr digest pin in .env.
    img="$(sed -n 's/^IMAGE=//p' .env | tail -1 | tr -d '"'"'"'"' | tr -d '[:space:]')"
    [ -n "$img" ] || img="$(grep -oE 'ghcr.io[^"'"'"']*sha256:[0-9a-f]{64}' .env | head -1)"
    if [ -n "$img" ]; then
        wipe_ok=1
        docker run --rm --entrypoint /bin/bash -v "$HOME/.cache/vllm-glm53-flash:/c" "$img"             -c 'rm -rf /c/triton /c/tilelang' || { echo "[prod-start] WARN: head cache wipe failed"; wipe_ok=0; }
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH"             "docker run --rm --entrypoint /bin/bash -v \$HOME/.cache/vllm-glm53-flash:/c '$img' -c 'rm -rf /c/triton /c/tilelang'"             || { echo "[prod-start] WARN: worker cache wipe failed"; wipe_ok=0; }
        # Advance the stamp ONLY when both nodes were wiped; a half-wipe must
        # retry on the next start (one rank on stale kernels is the 0.96->0.58 class).
        if [ "$wipe_ok" = 1 ]; then
            mkdir -p "$(dirname "$stamp")" && printf '%s\n' "$shape_hash" > "$stamp"
        else
            echo "[prod-start] WARN: JIT cache wipe incomplete — stamp left unchanged, will retry next start"
        fi
    else
        # Do NOT advance the stamp: a missed wipe must retry on the next start.
        echo "[prod-start] WARN: could not resolve IMAGE from .env — JIT caches NOT wiped, stamp left unchanged"
    fi
fi

./start.sh start
rc=$?
# LOCAL: content-prefix warmup — replay standing clients' anchors (local/warmup-anchors/)
# through the normal prefill path once the pair is ready. Best-effort: an empty or
# missing anchor dir is a no-op, and a warmup failure never fails the unit.
if [ "$rc" -eq 0 ] && [ -x "$(command -v python3)" ]; then
    python3 local/prefix-warmup.py || echo "[prod-start] WARN: prefix warmup failed (non-fatal)"
fi
exit "$rc"
