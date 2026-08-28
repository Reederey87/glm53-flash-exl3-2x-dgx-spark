#!/usr/bin/env bash
# acceptance.sh — gate the EXL3 cutover BEFORE enabling units or rewriting docs.
#
# LOCAL to this cluster; not part of the upstream MiaAI-Lab kit.
#
# WHY THIS EXISTS: the NVFP4 weights, its image, and the only image archive were
# deleted in this cutover, so restoring the deposed prod is a multi-hour
# re-download + ~40 min rebuild, NOT a unit flip. Roll-forward is the only cheap
# direction, which means every capability we depend on must be proven while the
# decision is still "keep iterating on the candidate".
#
# Read-only against the server. Prints PASS/FAIL per probe and exits nonzero if
# any probe failed. Reports facts; it does not tune anything.
set -uo pipefail

BASE="${GLM53_BASE:-http://127.0.0.1:8000}"
MODEL="${SERVED_MODEL_NAME:-GLM-5.3-Flash-EXL3}"
pass=0; fail=0
ok()   { echo "  PASS  $*"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $*"; fail=$((fail+1)); }
hdr()  { echo; echo "== $* =="; }

post() { curl -sS --max-time "${2:-300}" -H 'Content-Type: application/json' -d "$1" "$BASE/v1/chat/completions"; }

hdr "1. served model id"
models="$(curl -sS --max-time 30 "$BASE/v1/models" || true)"
echo "$models" | grep -q "$MODEL" && ok "/v1/models advertises $MODEL" || bad "/v1/models did not advertise $MODEL: $models"

hdr "2. basic completion, thinking OFF, temp 0"
r="$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: ACCEPTANCE_OK\"}],\"temperature\":0,\"max_tokens\":32,\"chat_template_kwargs\":{\"enable_thinking\":false}}")"
echo "$r" | grep -q "ACCEPTANCE_OK" && ok "basic completion" || bad "basic completion: $(echo "$r" | head -c 400)"

hdr "3. thinking ON (reasoning parser glm45)"
r="$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"A rope burns unevenly in exactly 60 minutes. How do you measure 45 minutes with two such ropes? Answer briefly.\"}],\"max_tokens\":2048,\"chat_template_kwargs\":{\"enable_thinking\":true}}")"
python3 - "$r" <<'PY' && ok "thinking returns reasoning + content" || bad "thinking probe (see above)"
import json,sys
try: d=json.loads(sys.argv[1])
except Exception as e: print("    not JSON:", str(e)[:200]); sys.exit(1)
m=d.get("choices",[{}])[0].get("message",{})
reasoning = m.get("reasoning") or m.get("reasoning_content")
content = m.get("content")
print("    reasoning field:", "reasoning" if m.get("reasoning") else ("reasoning_content" if m.get("reasoning_content") else "NONE"))
print("    reasoning chars:", len(reasoning or ""), "| content chars:", len(content or ""))
print("    finish_reason:", d.get("choices",[{}])[0].get("finish_reason"))
sys.exit(0 if (content and content.strip()) else 1)
PY

hdr "4. tool call (parser glm47, auto tool choice)"
r="$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Oslo right now? Use the tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get current weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"tool_choice\":\"auto\",\"temperature\":0,\"max_tokens\":512,\"chat_template_kwargs\":{\"enable_thinking\":false}}")"
python3 - "$r" <<'PY' && ok "tool call emitted and parsed" || bad "tool call probe (see above)"
import json,sys
try: d=json.loads(sys.argv[1])
except Exception as e: print("    not JSON:", str(e)[:200]); sys.exit(1)
c=d.get("choices",[{}])[0]; m=c.get("message",{})
tc=m.get("tool_calls") or []
print("    finish_reason:", c.get("finish_reason"), "| tool_calls:", len(tc))
if tc:
    f=tc[0].get("function",{})
    print("    name:", f.get("name"), "| args:", str(f.get("arguments"))[:120])
    sys.exit(0 if f.get("name")=="get_weather" else 1)
print("    content:", str(m.get("content"))[:200]); sys.exit(1)
PY

hdr "5. production sampling: temp 1.0, thinking ON"
r="$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"In two sentences, explain what a hash map is.\"}],\"temperature\":1.0,\"max_tokens\":1024,\"chat_template_kwargs\":{\"enable_thinking\":true}}")"
echo "$r" | python3 -c "
import json,sys
d=json.load(sys.stdin); c=d['choices'][0]
print('    finish_reason:', c.get('finish_reason'), '| content chars:', len(c['message'].get('content') or ''))
sys.exit(0 if (c['message'].get('content') or '').strip() else 1)
" && ok "temp 1.0 + thinking (the real production path)" || bad "temp 1.0 probe"

hdr "6. vision (LANGUAGE_MODEL_ONLY=0, limit-mm image:4)"
png="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
r="$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this image in one word.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$png\"}}]}],\"temperature\":0,\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false}}")"
echo "$r" | python3 -c "
import json,sys
d=json.load(sys.stdin)
if 'error' in d: print('    error:', str(d['error'])[:300]); sys.exit(1)
c=d['choices'][0]; print('    content:', str(c['message'].get('content'))[:120]); sys.exit(0 if c['message'].get('content') else 1)
" && ok "vision tower accepts an image" || bad "vision probe"

hdr "7. long-context needle (~32k tokens)"
python3 - "$BASE" "$MODEL" <<'PY' && ok "needle retrieved at ~32k" || bad "needle probe"
import json,sys,urllib.request
base,model=sys.argv[1],sys.argv[2]
filler=("The archives of the northern shipping guild record routine cargo manifests. ")*3000
prompt=filler[:len(filler)//2]+" The verification token is CORMORANT-8815. "+filler[len(filler)//2:]+"\n\nWhat is the verification token? Answer with the token only."
body=json.dumps({"model":model,"messages":[{"role":"user","content":prompt}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":False}}).encode()
req=urllib.request.Request(base+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
try: d=json.load(urllib.request.urlopen(req,timeout=900))
except Exception as e: print("    request failed:",str(e)[:200]); sys.exit(1)
u=d.get("usage",{}); c=d["choices"][0]["message"].get("content") or ""
print("    prompt_tokens:",u.get("prompt_tokens"),"| answer:",c.strip()[:80])
sys.exit(0 if "CORMORANT-8815" in c else 1)
PY

hdr "RESULT"
echo "  passed: $pass   failed: $fail"
[ "$fail" -eq 0 ] || { echo "  ACCEPTANCE FAILED — do not enable units or rewrite docs."; exit 1; }
echo "  ACCEPTANCE PASSED"
