#!/usr/bin/env bash
# watchdog.sh — health canary + auto-heal for the GLM-5.3-Flash EXL3 pair.
#
# LOCAL to this cluster; not part of the upstream MiaAI-Lab kit.
#
# DESIGN NOTE — why this is deliberately dumb:
#   The NVFP4 watchdog this replaces force-killed a HEALTHY engine because it
#   inferred saturation from metrics (kv_cache_usage >= 0.95 in watchdog.sh:143)
#   that GLM never reaches: it saturates on MAX_NUM_SEQS with KV at ~4%, so any
#   request queued behind the canary triggered a spurious pair bounce.
#   See glm53-flash-nvfp4/docs/BUG-watchdog-seq-saturation.md.
#
#   This one reads NO throughput or occupancy metrics. It answers two questions
#   only, both unambiguous:
#     1. Does the head answer /health?
#     2. Is the worker container actually running on spark2?
#   A busy engine still answers /health, so load can never be mistaken for death.
#
# Heals by restarting the systemd unit, which runs `start.sh stop` (removing
# BOTH containers) before `start.sh start` — i.e. an ordered pair-bounce, never
# a head-only restart.
set -uo pipefail

PORT="${PORT:-8000}"
UNIT="${UNIT:-vllm-glm53exl3.service}"
WORKER_SSH="${WORKER_SSH:-nvidia@192.168.177.11}"
WORKER_CONTAINER="${WORKER_CONTAINER:-glm53-exl3-worker}"
HEAD_CONTAINER="${HEAD_CONTAINER:-glm53-exl3-head}"

FAIL_THRESHOLD="${FAIL_THRESHOLD:-3}"      # consecutive bad ticks before healing
GRACE_SECS="${GRACE_SECS:-300}"            # post-start settle. NOT the cold-load window: the unit only
                                           # reaches "active" after start.sh already saw /health green,
                                           # so the slow load is covered by the "activating" branch above.
MIN_HEAL_INTERVAL="${MIN_HEAL_INTERVAL:-1800}"  # never bounce more than once per 30 min
CURL_TIMEOUT="${CURL_TIMEOUT:-10}"
STALL_THRESHOLD="${STALL_THRESHOLD:-10}"   # consecutive ticks of zero forward progress WHILE work is pending
LOAD_GRACE="${LOAD_GRACE:-4800}"           # a running container younger than this with /health down is loading,
                                           # not wedged. Matches READY_TIMEOUT in the unit.

STATE_DIR="${STATE_DIR:-$HOME/.local/state/glm53exl3-watchdog}"
mkdir -p "$STATE_DIR"
FAIL_FILE="$STATE_DIR/consecutive_failures"
PROGRESS_FILE="$STATE_DIR/last_progress_tokens"
STALL_FILE="$STATE_DIR/consecutive_stalls"
HEAL_FILE="$STATE_DIR/last_heal_epoch"

log() { echo "[glm53exl3-watchdog] $*"; }

now=$(date +%s)

# --- never act while the unit is mid-start, or inside its warmup grace --------
state="$(systemctl --user is-active "$UNIT" 2>/dev/null || true)"
if [ "$state" = "activating" ]; then
    log "unit is activating (cold load in progress) — standing down"
    exit 0
fi
if [ "$state" != "active" ]; then
    log "unit is '$state', not active — not ours to heal (someone stopped it deliberately)"
    echo 0 > "$FAIL_FILE"
    exit 0
fi

# Seconds since the unit last entered active state.
# Monotonic (usec since boot) — no locale/format parsing, unlike ActiveEnterTimestamp.
enter_us="$(systemctl --user show "$UNIT" -p ActiveEnterTimestampMonotonic --value 2>/dev/null || true)"
uptime_s="$(cut -d. -f1 /proc/uptime 2>/dev/null || true)"
if [[ "$enter_us" =~ ^[0-9]+$ ]] && [[ "$uptime_s" =~ ^[0-9]+$ ]] && [ "$enter_us" -gt 0 ]; then
    active_for=$(( uptime_s - enter_us / 1000000 ))
    if [ "$active_for" -lt "$GRACE_SECS" ]; then
        log "within ${GRACE_SECS}s post-start grace (${active_for}s in) — standing down"
        exit 0
    fi
else
    # Fail SAFE: if we cannot establish how long the unit has been up, do NOT heal.
    log "cannot read ActiveEnterTimestampMonotonic — standing down rather than risk a spurious bounce"
    exit 0
fi

# --- the two checks ----------------------------------------------------------
bad=""

# Container state is checked BEFORE /health, because it is the only thing that
# can tell a deliberate teardown apart from a crash. Checking /health first
# collapses both into "not answering" and makes the watchdog fight its operator:
#   * `./start.sh stop` leaves the unit active (RemainAfterExit) with the
#     containers removed -> we would restart what a human just stopped.
#   * `./start.sh restart` keeps the unit active for the whole 30-60 min reload
#     while /health is down -> we would `docker rm -f` the half-loaded pair
#     roughly three minutes in, every single time.
# docker gives an unambiguous three-way answer, so use it.
head_state="$(docker inspect -f '{{.State.Running}}' "$HEAD_CONTAINER" 2>/dev/null | tr -d '[:space:]')"
[ -z "$head_state" ] && head_state="ABSENT"

if [ "$head_state" = "ABSENT" ]; then
    # Nothing removes this container except `docker rm` - there is no --rm and
    # vLLM is PID 1, so a crash EXITS it but leaves it present. Absent therefore
    # means a human ran `start.sh stop` (or a stop/restart is in flight).
    log "head container is ABSENT — deliberate teardown (start.sh stop/restart), standing down."
    log "  production is managed via systemctl; to disarm for a window:"
    log "  systemctl --user stop vllm-glm53exl3-watchdog.timer"
    echo 0 > "$FAIL_FILE"; echo 0 > "$STALL_FILE"; rm -f "$PROGRESS_FILE"
    exit 0
fi

if [ "$head_state" != "true" ]; then
    # Present but exited: vLLM is the container's PID 1, so this is an engine
    # crash and nothing cleaned up after it. Unambiguous.
    bad="head container ${HEAD_CONTAINER} exited (State.Running=${head_state})"
else
    # Running. Distinguish "still loading" from "wedged" by container age: a
    # fresh container is a reload in progress (this is the manual-restart case).
    started_at="$(docker inspect -f '{{.State.StartedAt}}' "$HEAD_CONTAINER" 2>/dev/null || true)"
    started_epoch=$(date -d "$started_at" +%s 2>/dev/null || echo 0)
    age=$(( now - started_epoch ))
    if ! curl -fsS --max-time "$CURL_TIMEOUT" "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        if [ "$started_epoch" -le 0 ]; then
            # Fail SAFE: without a container age we cannot tell loading from wedged.
            log "cannot parse container StartedAt ('$started_at') — standing down rather than risk a spurious bounce"
            exit 0
        fi
        if [ "$age" -lt "$LOAD_GRACE" ]; then
            log "head container is only ${age}s old and /health is not up yet — load in progress, standing down"
            echo 0 > "$FAIL_FILE"
            exit 0
        fi
        bad="head /health not answering (container up ${age}s)"
    fi
fi

if [ -z "$bad" ]; then
    # Catches the spark2-only-reboot case: the worker container is launched by a
    # one-shot `ssh docker run -d` with no --restart policy and no unit on spark2,
    # so a worker reboot silently removes it while the head unit still reads active.
    #
    # Distinguish transport failure from container state: ssh exits 255 when it
    # cannot connect at all. A blip on rail 1 is not evidence the engine is dead
    # -- and if /health is green on a TP=2 pair, the worker is necessarily alive,
    # since collectives would otherwise hang.
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" \
        "docker inspect -f '{{.State.Running}}' '$WORKER_CONTAINER' 2>/dev/null | grep -q true" >/dev/null 2>&1
    rc=$?
    if [ "$rc" = "255" ]; then
        log "worker UNREACHABLE over ssh (transport, rc=255) — not treating as container death"
    elif [ "$rc" != "0" ]; then
        bad="worker container ${WORKER_CONTAINER} not running on ${WORKER_SSH}"
    fi
fi

# --- third signal: forward progress ------------------------------------------
# WHY THIS EXISTS: /health is NOT a sufficient liveness signal. In vLLM V1,
# AsyncLLM.check_health() only raises if self.errored is already set
# (vllm/v1/engine/async_llm.py:940) -- it never probes the workers. So /health
# returns 503 when the engine core PROCESS DIES, but stays 200 when the engine
# is WEDGED (a stuck NCCL collective, ranks alive but hung). That is the exact
# failure this pair is most exposed to across two nodes.
#
# WHY IT CANNOT REPEAT THE SEQ-SATURATION FALSE-KILL: it measures PROGRESS, not
# latency and not occupancy. A saturated engine still increments token counters,
# so "busy" and "wedged" are unambiguous. We use prompt+generation tokens summed,
# so a long 900k chunked prefill (which emits no generation tokens for minutes)
# still counts as progress. And we only look at all when work is actually
# pending -- an idle server is never stalled.
if [ -z "$bad" ]; then
    metrics="$(curl -fsS --max-time "$CURL_TIMEOUT" "http://127.0.0.1:${PORT}/metrics" 2>/dev/null || true)"
    if [ -n "$metrics" ]; then
        pending="$(awk '/^vllm:num_requests_(running|waiting)[ {]/ {s+=$NF} END {print s+0}' <<< "$metrics")"
        progress="$(awk '/^vllm:(prompt|generation)_tokens(_total)?[ {]/ {s+=$NF} END {printf "%.0f", s+0}' <<< "$metrics")"
        prev_progress="$(cat "$PROGRESS_FILE" 2>/dev/null || echo -1)"
        echo "$progress" > "$PROGRESS_FILE"

        if [ "${pending:-0}" -gt 0 ] && [ "$prev_progress" != "-1" ] && [ "$progress" = "$prev_progress" ]; then
            stalls=$(( $(cat "$STALL_FILE" 2>/dev/null || echo 0) + 1 ))
            echo "$stalls" > "$STALL_FILE"
            log "no forward progress ($stalls/$STALL_THRESHOLD) with ${pending} request(s) pending; tokens frozen at ${progress}"
            [ "$stalls" -ge "$STALL_THRESHOLD" ] && bad="engine wedged: ${pending} request(s) pending, zero tokens for ${stalls} ticks (/health still 200)"
        else
            echo 0 > "$STALL_FILE"
        fi
    fi
fi

if [ -z "$bad" ]; then
    prev="$(cat "$FAIL_FILE" 2>/dev/null || echo 0)"
    [ "$prev" != "0" ] && log "recovered after $prev consecutive failure(s)"
    echo 0 > "$FAIL_FILE"
    exit 0
fi

fails=$(( $(cat "$FAIL_FILE" 2>/dev/null || echo 0) + 1 ))
echo "$fails" > "$FAIL_FILE"
log "UNHEALTHY ($fails/$FAIL_THRESHOLD): $bad"
[ "$fails" -ge "$FAIL_THRESHOLD" ] || exit 0

# --- heal --------------------------------------------------------------------
last_heal="$(cat "$HEAL_FILE" 2>/dev/null || echo 0)"
if [ $((now - last_heal)) -lt "$MIN_HEAL_INTERVAL" ]; then
    log "last heal was $((now - last_heal))s ago (< ${MIN_HEAL_INTERVAL}s) — backing off, NOT bouncing."
    log "the pair is failing repeatedly; this needs a human. journalctl --user -u $UNIT"
    exit 0
fi

log "healing: ordered pair-bounce via 'systemctl --user restart $UNIT'"
echo "$now" > "$HEAL_FILE"
echo 0 > "$FAIL_FILE"
echo 0 > "$STALL_FILE"
rm -f "$PROGRESS_FILE"
if systemctl --user restart --no-block "$UNIT"; then
    # --no-block is REQUIRED: the target unit takes up to TimeoutStartSec=5400 to
    # come back, but this service is capped at 600s and would be SIGKILLed mid-bounce.
    # Honesty: --no-block returns success once the job is ENQUEUED. If StartLimit
    # refuses the start asynchronously the unit just sits failed, so do not claim
    # the heal happened -- say what we actually know and point at the check.
    log "restart ENQUEUED (--no-block; StartLimit may still refuse it)"
    log "  confirm with: systemctl --user is-active $UNIT"
else
    log "RESTART FAILED — needs a human"
fi
