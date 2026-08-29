#!/usr/bin/env python3
# LOCAL to this cluster; not part of the vendored upstream kit.
#
# TTFT / prefill probe: send a ~N-token prompt of RANDOM digits (defeats
# prefix caching, which is on) and measure time-to-first-token via SSE.
#   python3 ttft-probe.py <approx_tokens> [runs] [base_url]
# Prints one line per run: prompt_tokens, ttft_s, prefill_tok_s.
import json, random, sys, time, urllib.request

n = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8000"

for r in range(runs):
    rng = random.Random()
    # ~1.5 tokens per "NNNN " group on this tokenizer; oversample then trust usage.
    body = " ".join(str(rng.randint(1000, 9999)) for _ in range(int(n / 1.5)))
    prompt = f"Ignore the digits and reply with the single word OK.\n{body}\nReply OK."
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps({
            "model": "GLM-5.3-Flash-EXL3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic(); ttft = None; ptok = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if ttft is None and chunk.get("choices") and (
                    chunk["choices"][0].get("delta", {}).get("content")
                    or chunk["choices"][0].get("delta", {}).get("reasoning")):
                ttft = time.monotonic() - t0
            if chunk.get("usage"):
                ptok = chunk["usage"]["prompt_tokens"]
    print(f"run={r} prompt_tokens={ptok} ttft_s={ttft:.2f} "
          f"prefill_tok_s={ptok/ttft:.0f}")
