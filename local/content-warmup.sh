#!/usr/bin/env bash
# content-warmup.sh — replay standing clients' stable prompt anchors after a
# restart so their first real turn hits warm prefix-cache pages (W8).
#
# LOCAL to this cluster; not part of the upstream MiaAI-Lab kit.
#
# The kit's boot-shape-warmup burns KERNEL shapes; this warms CONTENT: each
# anchor goes through the ordinary prefill path as a normal request, landing
# real pages in the prefix cache (which sparse retention then keeps). Only
# byte-identical prefixes hit, so anchors must match what the client actually
# sends — capture them from real traffic, don't hand-write them.
#
# DORMANT unless GLM53_CONTENT_WARMUP=1. Anchors live in
#   ~/.local/share/glm53-content-warmup/
#     *.txt   — anchor text, replayed as the system message of a 1-token chat
#     *.json  — a full /v1/chat/completions request body, posted verbatim
#               (max_tokens is overridden to 1)
# Anchors under 3584 tokens (one hybrid page) can never hit — warned and sent
# anyway (harmless).
#
# Wire-up (at window time): systemd drop-in on vllm-glm53exl3.service —
#   [Service]
#   ExecStartPost=-/usr/bin/bash %h/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/local/content-warmup.sh
# The leading '-' keeps a warmup failure from failing the unit.
set -uo pipefail

[ "${GLM53_CONTENT_WARMUP:-0}" = "1" ] || { echo "content-warmup: disabled (GLM53_CONTENT_WARMUP != 1)"; exit 0; }

BASE="${GLM53_BASE:-http://127.0.0.1:8000}"
MODEL="${GLM53_MODEL:-GLM-5.3-Flash-EXL3}"
ANCHOR_DIR="${GLM53_ANCHOR_DIR:-$HOME/.local/share/glm53-content-warmup}"
REQ_TIMEOUT="${GLM53_WARMUP_REQ_TIMEOUT:-600}"   # bounded: a 100k cold prefill is ~2 min
PAGE_TOKENS=3584

ts() { date -u +%H:%M:%SZ; }
log() { echo "[content-warmup $(ts)] $*"; }

shopt -s nullglob
files=("$ANCHOR_DIR"/*.txt "$ANCHOR_DIR"/*.json)
[ "${#files[@]}" -gt 0 ] || { log "no anchors in $ANCHOR_DIR — nothing to warm"; exit 0; }

curl -sf -m 10 "$BASE/health" >/dev/null || { log "API not reachable at $BASE — skipping"; exit 0; }

warmed=0
for f in "${files[@]}"; do
    name="$(basename "$f")"
    case "$f" in
        *.json)
            body="$(python3 -c 'import json,sys; b=json.load(open(sys.argv[1])); b["max_tokens"]=1; b.setdefault("temperature",0); print(json.dumps(b))' "$f")" \
                || { log "SKIP $name: not valid JSON"; continue; }
            ;;
        *)
            # token-count check via /tokenize (best effort)
            n="$(python3 - "$BASE" "$f" <<'PY' 2>/dev/null
import json,sys,urllib.request
base,path=sys.argv[1],sys.argv[2]
req=urllib.request.Request(base+"/tokenize",data=json.dumps({"prompt":open(path).read()}).encode(),headers={"Content-Type":"application/json"})
print(json.load(urllib.request.urlopen(req,timeout=30)).get("count",0))
PY
)"
            [ -n "${n:-}" ] && [ "${n:-0}" -lt "$PAGE_TOKENS" ] 2>/dev/null \
                && log "WARN $name: ~${n} tokens < one ${PAGE_TOKENS}-token page — can never produce a hit"
            body="$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[2],"messages":[{"role":"system","content":open(sys.argv[1]).read()},{"role":"user","content":"OK"}],"max_tokens":1,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}))' "$f" "$MODEL")"
            ;;
    esac
    t0=$(date +%s)
    if out="$(curl -sf -m "$REQ_TIMEOUT" -H "Content-Type: application/json" -d "$body" "$BASE/v1/chat/completions" 2>&1)"; then
        cached="$(printf '%s' "$out" | python3 -c 'import json,sys; u=json.load(sys.stdin).get("usage") or {}; print((u.get("prompt_tokens_details") or {}).get("cached_tokens",0), u.get("prompt_tokens",0))' 2>/dev/null)"
        log "warmed $name in $(( $(date +%s) - t0 ))s (cached/prompt: ${cached:-?})"
        warmed=$((warmed+1))
    else
        log "WARN $name: request failed after $(( $(date +%s) - t0 ))s — continuing"
    fi
done
log "done: ${warmed}/${#files[@]} anchors warmed"
