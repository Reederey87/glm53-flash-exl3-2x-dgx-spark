#!/usr/bin/env bash
# Task 9 spec-graph probe. Read-only /metrics scrape plus live argv when
# Docker can see the head container. Distinguishes structured 1.000/7.000
# from vllm#53030 LENGTH=1 collapse. Does not restart, wipe JIT, or toggle
# ENFORCE_EAGER.
#
#   scripts/spec-graph-probe.sh [BASE]
#   BASE defaults to http://127.0.0.1:8000 (loopback). From the Mac use the
#   tunnel, typically http://127.0.0.1:18000. Mac tunnel runs do not invent
#   capture sizes or --enforce-eager; unknown argv withholds the eager arm.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
BASE="${1:-${GLM53_BASE:-http://127.0.0.1:8000}}"
HEAD_CONTAINER="${HEAD_CONTAINER:-glm53-exl3-head}"
CURL=(curl --noproxy '*' -fsS --max-time 10)
if [ -n "${VLLM_API_KEY:-}" ]; then
    CURL+=(-H "Authorization: Bearer ${VLLM_API_KEY}")
fi

metrics="$("${CURL[@]}" "${BASE%/}/metrics")"
argv_known=0
enforce=0
sizes=""
if command -v docker >/dev/null 2>&1 \
    && docker inspect -f '{{.State.Running}}' "$HEAD_CONTAINER" 2>/dev/null | grep -qx true; then
    argv="$(docker exec "$HEAD_CONTAINER" sh -c 'tr "\0" " " < /proc/1/cmdline' 2>/dev/null || true)"
    if [ -n "${argv:-}" ]; then
        argv_known=1
        case " $argv " in
            *" --enforce-eager "*) enforce=1 ;;
        esac
        captured="$(printf '%s\n' "$argv" | awk '
            {
                for (i = 1; i <= NF; i++) {
                    if ($i == "--cudagraph-capture-sizes") {
                        out = ""
                        for (j = i + 1; j <= NF && $j !~ /^-/; j++) {
                            out = out (out ? "," : "") $j
                        }
                        print out
                    }
                }
            }')"
        sizes="${captured:-}"
    fi
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
args=(
    "$PYTHON_BIN" "$ROOT/scripts/audit_spec_graph_probe.py"
    --min-drafts "${MIN_DRAFTS:-100}"
)
if [ "$argv_known" = 1 ]; then
    args+=(--argv-known)
    if [ -n "$sizes" ]; then
        args+=(--capture-sizes "$sizes")
    else
        args+=(--capture-sizes "")
    fi
    if [ "$enforce" = 1 ]; then
        args+=(--enforce-eager)
    fi
else
    args+=(--capture-sizes "")
fi
printf '%s\n' "$metrics" | "${args[@]}"
