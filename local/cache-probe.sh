#!/usr/bin/env bash
# cache-probe.sh — windowed prefix-cache hit-rate sampler, from the Mac,
# through the real access path (SSH port-forward -> spark1 loopback).
#
# LOCAL to this cluster; not part of the vendored upstream kit.
#
# The engine's logged "Prefix cache hit rate" and the dashboard's lifetime
# ratio hide short-term collapses: the counters are cumulative since boot.
# This prints the DELTA hit rate per sampling window, next to the KV-usage
# gauge and the scheduler queue, which is what you need to see eviction
# churn (hit% ~0 while waiting{capacity} > 0) as it happens.
#
# NOTE metric semantics: vllm:kv_cache_usage_perc counts only blocks held by
# RUNNING requests (it reads 0.0 at idle even with a warm cache). Cached
# reusable blocks live in the free pool and are invisible to that gauge.
# ⚠ The global queries/hits counters over-count 1.3–2x under queue pressure
# (vLLM RFC #37003 thread) — this probe watches OTHER clients' live traffic so
# they are the only option here; treat the windowed hit% as directional. For
# exact numbers use cache-burst.py, which reads per-request
# usage.prompt_tokens_details.cached_tokens from its own requests.
#
# Usage: cache-probe.sh [interval_seconds] [count]
set -uo pipefail

BASE="${GLM53_BASE:-http://127.0.0.1:18000}"   # launchd com.dgxspark.deepseek-tunnel
INTERVAL="${1:-30}"
COUNT="${2:-0}"        # 0 = run until interrupted

get() {
    curl -sS --max-time 10 "$BASE/metrics" 2>/dev/null \
      | awk '
        /^vllm:prefix_cache_queries_total/ {q=$2}
        /^vllm:prefix_cache_hits_total/    {h=$2}
        /^vllm:kv_cache_usage_perc/        {u=$2}
        /^vllm:num_requests_running/       {r=$2}
        /^vllm:num_requests_waiting_by_reason\{.*capacity/ {w=$2}
        /^vllm:prompt_tokens_total/        {p=$2}
        END {printf "%.0f %.0f %.4f %.0f %.0f %.0f\n", q, h, u, r, w, p}'
}

printf "%-9s %10s %10s %7s %6s %8s %10s\n" \
    "time" "Δqueries" "Δhits" "hit%" "kv%" "run/wait" "prompt_tps"
prev=""
i=0
while :; do
    now="$(get)" || { echo "metrics fetch failed"; sleep "$INTERVAL"; continue; }
    if [ -n "$now" ] && [ -n "$prev" ]; then
        read -r q1 h1 _ _ _ p1 <<<"$prev"
        read -r q2 h2 u r w p2 <<<"$now"
        dq=$((q2-q1)); dh=$((h2-h1)); dp=$((p2-p1))
        if [ "$dq" -gt 0 ]; then hr=$(awk -v a="$dh" -v b="$dq" 'BEGIN{printf "%.1f", 100*a/b}'); else hr="-"; fi
        printf "%-9s %10s %10s %7s %5.1f%% %8s %10s\n" \
            "$(date +%H:%M:%S)" "$dq" "$dh" "$hr" \
            "$(awk -v u="$u" 'BEGIN{print 100*u}')" "$r/$w" \
            "$(awk -v p="$dp" -v s="$INTERVAL" 'BEGIN{printf "%.0f", p/s}')"
    fi
    prev="$now"
    i=$((i+1))
    [ "$COUNT" -gt 0 ] && [ "$i" -gt "$COUNT" ] && break
    sleep "$INTERVAL"
done
