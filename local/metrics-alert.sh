#!/usr/bin/env bash
# LOCAL to this cluster; not part of the upstream MiaAI-Lab kit.
#
# Alert-only metrics canary for GLM-5.3-Flash-EXL3. Runs from a 5-min user
# timer on spark1. NEVER heals — healing belongs to the watchdog, which is
# deliberately dumb; this script watches the two things the watchdog
# deliberately does not:
#
#   1. DFlash2 draft acceptance collapse. The whole performance story of this
#      deployment is the k=7 drafter (structured 67 vs ~24 tok/s baseline). A
#      silent acceptance collapse (bad template change, drafter mis-load)
#      would look "healthy" to every liveness probe. Interval acceptance
#      below ACCEPT_WARN with at least MIN_DRAFT drafted tokens => exit 1
#      (unit goes failed => visible in `systemctl --user --failed`).
#   2. Live-argv integrity. The serve command is built by generated inner
#      scripts; a stale script or leftover env is a silent config-drift path.
#      We read the REAL argv of PID 1 in the head container and assert the
#      config-critical flags are present.
#
# State: interval counters in $STATE_DIR. First run after boot only seeds
# state. A down/loading server is the watchdog's problem — stand down (exit 0).
set -uo pipefail
PORT="${PORT:-8000}"
HEAD_CONTAINER="${HEAD_CONTAINER:-glm53-exl3-head}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/glm53exl3-metrics}"
ACCEPT_WARN="${ACCEPT_WARN:-0.30}"
MIN_DRAFT="${MIN_DRAFT:-50}"
# Config-critical argv fragments (pipe-separated). Every one must be present.
# 900000 -> 1000000 2026-08-29 (the 1M Window-2 geometry; the stale value had the
# canary failing every tick). Keep this in lockstep with MAX_MODEL_LEN in .env.
EXPECT_ARGV="${EXPECT_ARGV:---host 127.0.0.1|--port ${PORT}|--kv-cache-memory-bytes 15414698763|--quantization exl3|--max-model-len 1000000}"

mkdir -p "$STATE_DIR"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
problems=0

# --- stand down unless the server is actually up --------------------------
running="$(docker inspect -f '{{.State.Running}}' "$HEAD_CONTAINER" 2>/dev/null | tr -d '[:space:]')"
if [ "$running" != "true" ]; then
  echo "$(ts) metrics-alert: head container not running — standing down (watchdog's problem)"
  exit 0
fi
if ! curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "$(ts) metrics-alert: /health not answering — standing down (loading or watchdog's problem)"
  exit 0
fi

# --- 1. live-argv integrity ----------------------------------------------
argv="$(docker exec "$HEAD_CONTAINER" sh -c 'tr "\0" " " < /proc/1/cmdline' 2>/dev/null)"
if [ -z "$argv" ]; then
  echo "$(ts) metrics-alert: could not read container PID-1 argv — skipping integrity check"
else
  missing=""
  IFS='|' read -r -a frags <<< "$EXPECT_ARGV"
  for f in "${frags[@]}"; do
    case "$argv" in *"$f"*) ;; *) missing="$missing [$f]";; esac
  done
  if [ -n "$missing" ]; then
    echo "$(ts) metrics-alert: *** ARGV INTEGRITY FAIL — live serve command is missing:$missing"
    echo "$(ts) metrics-alert: live argv: $argv"
    problems=$((problems+1))
  fi
fi

# --- 2. DFlash2 acceptance over the interval ------------------------------
metrics="$(curl -sf -m 10 "http://127.0.0.1:${PORT}/metrics" 2>/dev/null)"
if [ -z "$metrics" ]; then
  echo "$(ts) metrics-alert: /metrics scrape failed — skipping acceptance check"
else
  drafted="$(awk '/^vllm:spec_decode_num_draft_tokens(_total)?[ {]/ {s+=$NF} END {printf "%.0f", s+0}' <<< "$metrics")"
  accepted="$(awk '/^vllm:spec_decode_num_accepted_tokens(_total)?[ {]/ {s+=$NF} END {printf "%.0f", s+0}' <<< "$metrics")"
  state="$STATE_DIR/spec_counters"
  prev_d=0; prev_a=0
  if [ -f "$state" ]; then read -r prev_d prev_a < "$state" 2>/dev/null || true; fi
  printf '%s %s\n' "$drafted" "$accepted" > "$state.tmp" && mv "$state.tmp" "$state"
  d=$((drafted - prev_d)); a=$((accepted - prev_a))
  if [ "$d" -lt 0 ]; then
    echo "$(ts) metrics-alert: counters went backwards (server restarted) — reseeded state"
  elif [ "$d" -ge "$MIN_DRAFT" ]; then
    ratio="$(awk -v a="$a" -v d="$d" 'BEGIN{printf "%.4f", a/d}')"
    if awk -v r="$ratio" -v w="$ACCEPT_WARN" 'BEGIN{exit !(r<w)}'; then
      echo "$(ts) metrics-alert: *** DFLASH2 ACCEPTANCE COLLAPSE: $ratio over interval ($a/$d) < $ACCEPT_WARN"
      echo "$(ts) metrics-alert: check template/drafter changes; structured baseline is ~0.92-0.98"
      problems=$((problems+1))
    else
      echo "$(ts) metrics-alert: acceptance $ratio ($a/$d) — ok"
    fi
  else
    echo "$(ts) metrics-alert: interval too thin ($d drafted < $MIN_DRAFT) — no verdict"
  fi
fi

exit $((problems > 0 ? 1 : 0))
