#!/usr/bin/env python3
"""toolcall-probe.py — blank-required-args prober for upstream kit issue #10, from the Mac.

LOCAL to this cluster; not part of the vendored upstream kit.

Fires multi-tool-call turns (two tools, required string args) at the server under
several load shapes — solo, homogeneous concurrency, and mixed with large cold-prefill
peers — and checks every returned tool call for blank/missing required arguments.
Streaming responses are SSE-reconstructed the way real clients see them.

Usage:
  uv run python local/toolcall-probe.py               # full battery
  uv run python local/toolcall-probe.py --quick       # skip the mixed cold-pad phases
"""
import argparse, json, random, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

TOOLS = [
    {"type": "function", "function": {
        "name": "terminal_execute",
        "description": "Run a shell command on the host.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "browser_action",
        "description": "Drive the browser.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"},
                                      "url": {"type": "string"}},
                       "required": ["command", "url"]}}},
]
REQUIRED = {"terminal_execute": ["command"], "browser_action": ["command", "url"]}
PROMPT = ("Call BOTH tools exactly once in this turn: run `echo hello-issue10` in the "
          "terminal, and open https://example.com/issue10 in the browser. Use the tools; "
          "do not answer in prose.")

def make_pad(seed, words):
    rng = random.Random(seed)
    w = "ship hull engine cargo route port fuel speed wave tide deck crew".split()
    return " ".join(rng.choice(w) for _ in range(words))

def sse_collect(resp):
    """Reconstruct message + tool_calls from an SSE stream like a real client."""
    calls, finish = {}, None
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            d = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        ch = (d.get("choices") or [{}])[0]
        finish = ch.get("finish_reason") or finish
        for tc in (ch.get("delta") or {}).get("tool_calls") or []:
            i = tc.get("index", 0)
            slot = calls.setdefault(i, {"name": "", "args": ""})
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] += fn["name"]
            if fn.get("arguments"):
                slot["args"] += fn["arguments"]
    return calls, finish

def one_request(base, model, body_extra, stream, timeout):
    body = {"model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "tools": TOOLS, "tool_choice": "auto",
            "max_tokens": 512, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(body_extra)
    req = urllib.request.Request(f"{base}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if stream:
                calls, finish = sse_collect(r)
                calls = list(calls.values())
            else:
                d = json.load(r)
                ch = d["choices"][0]
                finish = ch.get("finish_reason")
                calls = [{"name": (tc.get("function") or {}).get("name", ""),
                          "args": (tc.get("function") or {}).get("arguments", "")}
                         for tc in (ch["message"].get("tool_calls") or [])]
    except Exception as e:
        return {"outcome": "error", "err": str(e)[:120], "wall": time.time() - t0}
    bad = []
    for c in calls:
        try:
            args = json.loads(c["args"]) if c["args"] else {}
        except json.JSONDecodeError:
            bad.append(f"{c['name']}: unparseable args {c['args']!r:.80}")
            continue
        for k in REQUIRED.get(c["name"], []):
            if not str(args.get(k, "")).strip():
                bad.append(f"{c['name']}: blank/missing required '{k}' (args={args})")
    outcome = "BAD_ARGS" if bad else ("ok" if len(calls) >= 2 else f"short({len(calls)} calls)")
    return {"outcome": outcome, "bad": bad, "finish": finish,
            "n_calls": len(calls), "wall": time.time() - t0}

def pad_request(base, model, seed, words, timeout):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user",
                                     "content": make_pad(seed, words) + " Reply with one word."}],
                       "temperature": 0, "max_tokens": 16,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            json.load(r)
        return {"outcome": "pad-ok", "wall": time.time() - t0}
    except Exception as e:
        return {"outcome": "pad-error", "err": str(e)[:120], "wall": time.time() - t0}

def phase(name, jobs, workers):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(*j) for j in jobs]
        res = [f.result() for f in futs]
    tool_res = [r for r in res if not r["outcome"].startswith("pad")]
    bad = [r for r in tool_res if r["outcome"] == "BAD_ARGS"]
    errs = [r for r in res if r["outcome"] in ("error", "pad-error")]
    print(f"{name:44s} n={len(tool_res):2d} bad_args={len(bad)} "
          f"short={sum(1 for r in tool_res if r['outcome'].startswith('short'))} "
          f"errors={len(errs)} wall={time.time()-t0:6.1f}s")
    for r in bad:
        print(f"    BAD (finish={r['finish']}): {r['bad']}")
    for r in errs[:3]:
        print(f"    err: {r.get('err')}")
    return len(tool_res), len(bad)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18000")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--pad-words", type=int, default=60000)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    T, M, TO = (a.base,), a.model, a.timeout
    total = bad = 0

    def tc(stream, temp):
        return (one_request, a.base, M, {"temperature": temp}, stream, TO)

    n, b = phase("c=1 stream temp0 x3", [tc(True, 0)] * 3, 1); total += n; bad += b
    n, b = phase("c=1 non-stream temp0 x2", [tc(False, 0)] * 2, 1); total += n; bad += b
    n, b = phase("c=1 stream temp1 (prod sampling) x3", [tc(True, 1.0)] * 3, 1); total += n; bad += b
    n, b = phase("c=4 homogeneous stream temp1 x8",
                 [tc(True, 1.0)] * 8, 4); total += n; bad += b
    if not a.quick:
        seedbase = int(time.time())  # fresh pads every run: cold prefill guaranteed
        for wave in range(2):
            jobs = [(pad_request, a.base, M, seedbase + wave * 2, a.pad_words, TO),
                    (pad_request, a.base, M, seedbase + wave * 2 + 1, a.pad_words, TO),
                    tc(True, 1.0), tc(True, 1.0)]
            n, b = phase(f"c=4 mixed: 2 cold ~{a.pad_words} pads + 2 tool (wave {wave+1})",
                         jobs, 4); total += n; bad += b
        jobs = [(pad_request, a.base, M, seedbase + 100, a.pad_words, TO)] + [tc(True, 1.0)] * 3
        n, b = phase("c=4 mixed: 1 cold pad + 3 tool", jobs, 4); total += n; bad += b
    print(f"\nTOTAL tool turns: {total}, blank-required-arg turns: {bad}")

if __name__ == "__main__":
    main()
