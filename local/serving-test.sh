#!/usr/bin/env bash
# serving-test.sh — exercise the EXL3 deployment AS A CLIENT, from the Mac,
# through the real access path (SSH port-forward -> spark1 loopback).
#
# LOCAL to this cluster; not part of the vendored upstream kit.
#
# The acceptance battery runs ON spark1 against 127.0.0.1 and proves the model
# works. This proves the SERVICE works: the tunnel, streaming, concurrency, and
# the OpenAI surface real clients actually use. Run it from the Mac.
set -uo pipefail

BASE="${GLM53_BASE:-http://127.0.0.1:18000}"   # launchd com.dgxspark.deepseek-tunnel
MODEL="${SERVED_MODEL_NAME:-GLM-5.3-Flash-EXL3}"
pass=0; fail=0
ok()  { echo "  PASS  $*"; pass=$((pass+1)); }
bad() { echo "  FAIL  $*"; fail=$((fail+1)); }
hdr() { echo; echo "== $* =="; }

hdr "1. tunnel reaches the API"
m="$(curl -sS --max-time 30 "$BASE/v1/models" 2>&1)"
if echo "$m" | grep -q "$MODEL"; then ok "tunnel -> $MODEL"; else bad "tunnel: $(echo "$m" | head -c 300)"; fi

hdr "2. non-streaming chat completion"
r="$(curl -sS --max-time 300 -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: SERVING_OK\"}],\"temperature\":0,\"max_tokens\":32,\"chat_template_kwargs\":{\"enable_thinking\":false}}" "$BASE/v1/chat/completions")"
if echo "$r" | grep -q "SERVING_OK"; then ok "non-streaming"; else bad "non-streaming: $(echo "$r" | head -c 300)"; fi

hdr "3. STREAMING (SSE) — the path most clients use"
tmp=$(mktemp)
curl -sSN --max-time 300 -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 30 separated by spaces.\"}],\"temperature\":0,\"max_tokens\":200,\"stream\":true,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  "$BASE/v1/chat/completions" > "$tmp" 2>&1
chunks=$(grep -c '^data: ' "$tmp" 2>/dev/null || echo 0)
done_ok=$(grep -c '^data: \[DONE\]' "$tmp" 2>/dev/null || echo 0)
text=$(python3 - "$tmp" <<'PY'
import json,sys
out=[]
for line in open(sys.argv[1]):
    if not line.startswith("data: ") or line.strip()=="data: [DONE]": continue
    try: d=json.loads(line[6:])
    except Exception: continue
    for c in d.get("choices",[]):
        out.append((c.get("delta") or {}).get("content") or "")
print("".join(out)[:80])
PY
)
echo "    SSE chunks: $chunks | [DONE]: $done_ok | text: $text"
if [ "$chunks" -gt 5 ] && [ "$done_ok" -ge 1 ]; then ok "streaming works ($chunks chunks, terminated)"; else bad "streaming (chunks=$chunks done=$done_ok)"; fi
rm -f "$tmp"

hdr "4. concurrency — 4 simultaneous (MAX_NUM_SEQS=4)"
t0=$(date +%s)
pids=(); outs=()
for i in 1 2 3 4; do
  o=$(mktemp); outs+=("$o")
  curl -sS --max-time 600 -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one sentence about the number $i. Then say DONE$i.\"}],\"temperature\":0,\"max_tokens\":200,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    "$BASE/v1/chat/completions" > "$o" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
t1=$(date +%s)
# Assert what concurrency actually has to prove: every request completes, and no
# two responses are the same (i.e. no cross-talk between in-flight sequences).
# Do NOT assert an exact marker string -- an earlier version required "DONE$i"
# and failed a perfectly healthy run because greedy decoding emitted "DONE"
# without the digit on one of four otherwise-correct, distinct answers.
good=$(python3 - "${outs[@]}" <<'PY'
import json, sys
contents = []
for path in sys.argv[1:]:
    try:
        d = json.load(open(path))
        c = d["choices"][0]
        m = (c["message"].get("content") or "").strip()
        if c.get("finish_reason") == "stop" and m:
            contents.append(m)
    except Exception:
        pass
distinct = len(set(contents))
print(distinct if distinct == len(contents) else 0)
PY
)
echo "    $good/4 completed with pairwise-distinct content in $((t1-t0))s"
if [ "$good" -eq 4 ]; then ok "4-way concurrency, no cross-talk"; else bad "concurrency: $good/4"; fi
rm -f "${outs[@]}"

hdr "5. tool call through the tunnel"
r="$(curl -sS --max-time 300 -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Weather in Bergen? Use the tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"tool_choice\":\"auto\",\"temperature\":0,\"max_tokens\":256,\"chat_template_kwargs\":{\"enable_thinking\":false}}" "$BASE/v1/chat/completions")"
if echo "$r" | python3 -c "
import json,sys
d=json.load(sys.stdin); c=d['choices'][0]; tc=c['message'].get('tool_calls') or []
print('    finish:', c.get('finish_reason'), '| calls:', len(tc), ('| '+str(tc[0]['function']['name'])+' '+str(tc[0]['function']['arguments'])[:60]) if tc else '')
sys.exit(0 if tc else 1)"; then ok "tool call over tunnel"; else bad "tool call over tunnel"; fi

hdr "6. sustained: 5 sequential requests, all must succeed"
# Assert what "sustained serving" actually means: every request completes with
# finish_reason=stop and non-empty content. Do NOT assert exact wording -- an
# earlier version grepped for "OK$i" and flagged a healthy reply as a failure
# because greedy decoding answered "Say OK2" with "OK". Cross-talk/distinctness
# is already covered by the concurrency probe above.
okc=0
for i in $(seq 1 5); do
  curl -sS --max-time 300 -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK$i\"}],\"temperature\":0,\"max_tokens\":16,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    "$BASE/v1/chat/completions" | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(1)
c=d.get('choices',[{}])[0]
sys.exit(0 if c.get('finish_reason')=='stop' and (c.get('message',{}).get('content') or '').strip() else 1)
" && okc=$((okc+1))
done
echo "    $okc/5 succeeded"
if [ "$okc" -eq 5 ]; then ok "sustained sequential"; else bad "sustained: $okc/5"; fi

hdr "RESULT"
echo "  passed: $pass   failed: $fail"
[ "$fail" -eq 0 ] || { echo "  SERVING TEST FAILED"; exit 1; }
echo "  SERVING TEST PASSED"
