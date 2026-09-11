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

log "validating configuration before stop or JIT-cache handling"
if ./start.sh validate; then
    :
else
    rc=$?
    log "configuration invalid — production left untouched"
    exit "$rc"
fi

log "stopping any running pair (idempotent)"
./start.sh stop || true

avail_gib() { # MemFree: the closest proxy to what CUDA reports as device-free.
    awk '/^MemFree:/ {printf "%d", $2/1048576}' /proc/meminfo
}
avail_gib_worker() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" \
        "awk '/^MemFree:/ {printf \"%d\", \$2/1048576}' /proc/meminfo" 2>/dev/null
}

# The JIT wipe only needs a container carrying /bin/bash and `rm`; it does NOT
# have to be the image being deployed. That distinction matters on the worker:
# start.sh ships the new tag to it AFTER this block runs, so requesting the new
# tag here always failed with
#     pull access denied for glm53-selfbuild
# and silently left a half-wipe (head wiped, worker not) while the stamp stayed
# unadvanced. Resolve an image the node actually has, preferring the requested
# tag. Echoes the tag, or nothing when the node has no glm53-selfbuild image.
wipe_image_for() { # $1 = "" for the head, else an ssh target
    local target="$1"
    if [ -z "$target" ]; then
        if docker image inspect "$img" >/dev/null 2>&1; then printf '%s' "$img"; return 0; fi
        docker images --format '{{.Repository}}:{{.Tag}}' | grep -E '^glm53-selfbuild:' | head -1
    else
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" \
            "if docker image inspect '$img' >/dev/null 2>&1; then printf '%s' '$img'; else docker images --format '{{.Repository}}:{{.Tag}}' | grep -E '^glm53-selfbuild:' | head -1; fi" 2>/dev/null
    fi
}

settle_wait() {
    log "waiting for >= ${NEED_GIB} GiB MemFree on BOTH nodes (timeout ${SETTLE_TIMEOUT}s)"
    local deadline h w
    deadline=$(( $(date +%s) + SETTLE_TIMEOUT ))
    while :; do
        h="$(avail_gib)"; w="$(avail_gib_worker)"
        if [[ "$h" =~ ^[0-9]+$ ]] && [[ "$w" =~ ^[0-9]+$ ]] \
           && [ "$h" -ge "$NEED_GIB" ] && [ "$w" -ge "$NEED_GIB" ]; then
            log "memory settled: head ${h} GiB, worker ${w} GiB"
            return 0
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            # Proceed anyway: vLLM's own pre-check is the real gate and will fail
            # cleanly with a precise number. Better that than silently never starting.
            log "WARN: timed out waiting to settle (head=${h:-?} worker=${w:-?} GiB, need ${NEED_GIB})"
            log "WARN: starting anyway — vLLM's pre-check will report the exact shortfall"
            return 1
        fi
        sleep "$SETTLE_INTERVAL"
    done
}

settle_wait || true
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
# GLM53_ADAPTIVE_K (policy) is NOT hashed: it trims spec_token_ids only.
# GLM53_ADAPTIVE_K_CAPTURE changes extra FULL graphs (task 25 B0/B).
# GLM53_KDA_REC_* change the captured fused_recurrent_kda Triton specialization
# (task 30). Unset = stock warps=1 / stages=3 / BV-cap=8.
shape_hash="$( { grep -E '^(DFLASH_TOKENS|DFLASH_DRAFT_TP|DFLASH_MODEL|DFLASH_REVISION|MTP_TOKENS|SPEC_METHOD|MAX_NUM_BATCHED_TOKENS|MAX_NUM_SEQS|MAX_MODEL_LEN|IMAGE|EXTRA_ARGS|GLM53_ADAPTIVE_K_CAPTURE|GLM53_KDA_REC_WARPS|GLM53_KDA_REC_STAGES|GLM53_KDA_REC_BV_CAP)=' .env 2>/dev/null; grep -E '^DFLASH_REVISION=' start.sh 2>/dev/null; printf 'GLM53_ADAPTIVE_K_CAPTURE=%s\n' "${GLM53_ADAPTIVE_K_CAPTURE:-}"; printf 'GLM53_KDA_REC_WARPS=%s\n' "${GLM53_KDA_REC_WARPS:-}"; printf 'GLM53_KDA_REC_STAGES=%s\n' "${GLM53_KDA_REC_STAGES:-}"; printf 'GLM53_KDA_REC_BV_CAP=%s\n' "${GLM53_KDA_REC_BV_CAP:-}"; } | sort | md5sum | cut -d' ' -f1)"
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
        head_img="$(wipe_image_for "")"
        worker_img="$(wipe_image_for "$WORKER_SSH")"
        if [ -n "$head_img" ]; then
            docker run --rm --entrypoint /bin/bash -v "$HOME/.cache/vllm-glm53-flash:/c" "$head_img" \
                -c 'rm -rf /c/triton /c/tilelang' \
                || { echo "[prod-start] WARN: head cache wipe failed"; wipe_ok=0; }
        else
            echo "[prod-start] WARN: head has no glm53-selfbuild image to wipe with"; wipe_ok=0
        fi
        if [ -n "$worker_img" ]; then
            ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" \
                "docker run --rm --entrypoint /bin/bash -v \$HOME/.cache/vllm-glm53-flash:/c '$worker_img' -c 'rm -rf /c/triton /c/tilelang'" \
                || { echo "[prod-start] WARN: worker cache wipe failed"; wipe_ok=0; }
        else
            echo "[prod-start] WARN: worker has no glm53-selfbuild image to wipe with"; wipe_ok=0
        fi
        # Advance the stamp ONLY when both nodes were wiped; a half-wipe must
        # retry on the next start (one rank on stale kernels is the 0.96->0.58 class).
        if [ "$wipe_ok" = 1 ]; then
            mkdir -p "$(dirname "$stamp")" && printf '%s\n' "$shape_hash" > "$stamp"
            echo "[prod-start] JIT caches wiped on both nodes (head=$head_img worker=$worker_img)"
        else
            echo "[prod-start] WARN: JIT cache wipe incomplete — stamp left unchanged, will retry next start"
        fi
    else
        # Do NOT advance the stamp: a missed wipe must retry on the next start.
        echo "[prod-start] WARN: could not resolve IMAGE from .env — JIT caches NOT wiped, stamp left unchanged"
    fi
fi

# --- start, with a bounded retry on the memory pre-check --------------------
# The settle gate above is deliberately below vLLM's own demand (0.85 x 121.69
# = 103.44 GiB) because idle MemFree sits at 93-97 GiB: page cache is reclaimed
# DURING the boot, so the node's free memory rises after the gate has already
# passed. Measured 2026-09-10 (task 35): the worker reported 103.09 GiB against
# a 103.44 GiB requirement and exited 1, a near miss that a second attempt a
# minute later cleared. Retrying is the correct fix — lowering the gate would
# convert a clean pre-check failure into a real OOM later.
#
# The retry is only correct for a TRANSIENT failure. Two defects measured
# 2026-09-11 make the naive loop worse than useless:
#
#  1. A failed attempt can leave a PARTIAL launch behind. `start.sh start` only
#     removes containers inside launch_cluster(), so a failure BEFORE that point
#     (preflight, ensure_image, download/sync of weights) leaves whatever the
#     previous attempt started still running — holding unified memory and the
#     API/master ports. The next attempt then fails preflight for a new reason,
#     and waiting for memory cannot recover it. Each retry must first tear the
#     pair down again.
#  2. A DETERMINISTIC failure (unresolvable RoCE GID, wrong fabric IP, RDMA port
#     down, worker unreachable) fails identically every time. Burning three
#     attempts and six seconds on it, while leaving containers up, produced the
#     confusing "3 attempts failed" report from a single configuration problem.
#
# So: tear down, then re-check the deterministic preconditions. If they now
# fail, abort immediately with that reason. Otherwise wait and retry.
MAX_BOOT_ATTEMPTS="${MAX_BOOT_ATTEMPTS:-3}"
# Validate the bound. `[ "$attempt" -ge "$MAX_BOOT_ATTEMPTS" ]` is false for a
# non-numeric value, so a typo'd MAX_BOOT_ATTEMPTS would disable the bound
# entirely and spin forever on a persistent failure that still passes preflight.
case "$MAX_BOOT_ATTEMPTS" in
    ''|*[!0-9]*)
        log "WARN: MAX_BOOT_ATTEMPTS='$MAX_BOOT_ATTEMPTS' is not a positive integer — using 3"
        MAX_BOOT_ATTEMPTS=3 ;;
    *)
        [ "$MAX_BOOT_ATTEMPTS" -ge 1 ] || {
            log "WARN: MAX_BOOT_ATTEMPTS=$MAX_BOOT_ATTEMPTS is below 1 — using 3"
            MAX_BOOT_ATTEMPTS=3
        } ;;
esac
attempt=0
while :; do
    attempt=$((attempt + 1))
    log "starting pair (attempt ${attempt}/${MAX_BOOT_ATTEMPTS})"
    if ./start.sh start; then
        exit 0
    else
        # Capture the failure status HERE. `rc=$?` after the completed `if`
        # would read the if-statement's own status, which is 0 when the
        # condition failed and no branch ran — so exhausting the retries would
        # have exited 0 with production down and masked the failure from any
        # supervisor.
        rc=$?
    fi

    # Tear down whatever the failed attempt left running BEFORE deciding
    # anything else. This must happen on the FINAL attempt too: start.sh only
    # removes containers inside launch_cluster(), so a failure earlier in the
    # boot leaks them, and they hold unified memory plus the API/master ports.
    # Exiting without cleanup reported "production left down" while leaving the
    # wreckage in place, which is what made the next manual start fail.
    log "start failed (rc=$rc) — cleaning up any partial launch"
    ./start.sh stop || log "WARN: cleanup stop returned non-zero; continuing"

    if [ "$attempt" -ge "$MAX_BOOT_ATTEMPTS" ]; then
        log "ERROR: start failed after ${attempt} attempt(s) — production left down, see the log above"
        exit "$rc"
    fi

    # Deterministic failure? A read-only preflight re-check answers this without
    # duplicating the fabric/GID logic here. If the environment is still bad,
    # another attempt cannot help — report the real reason and stop.
    if ! ./start.sh preflight >/dev/null 2>&1; then
        log "ERROR: preflight still fails — this is a configuration/environment problem, not a transient one"
        log "ERROR: not retrying (attempt ${attempt} of ${MAX_BOOT_ATTEMPTS} was the last to run); see the preflight output below"
        ./start.sh preflight || true
        exit "$rc"
    fi

    log "preflight passes — re-waiting for memory to settle, then retrying"
    settle_wait || true
done
