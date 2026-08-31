#!/usr/bin/env python3
"""W20 probe: per-stream decode at C1 vs C4 (kit issue #56 protocol).

N threads, one unique ~8k-token prompt each (unique salt so APC cannot serve one
stream from another), 400-token generations at temp 0, thinking off. Decode rate
is measured first-token-to-last from the SSE stream (prefill excluded). Prints
per-stream tok/s, the aggregate, and TTFT spread.

  uv run python local/w20-concurrency-probe.py --concurrency 1
  uv run python local/w20-concurrency-probe.py --concurrency 4
"""
import argparse, json, random, threading, time, urllib.request

MODEL = "GLM-5.3-Flash-EXL3"
WORDS = ["alpha", "beam", "cache", "delta", "ember", "fjord", "glyph", "hinge", "ionic",
         "joule", "kelvin", "lumen", "matrix", "nadir", "orbit", "prism", "quartz", "rotor",
         "sigma", "torus", "umbra", "vector", "wafer", "xenon", "yield", "zenith"]


def doc(seed, n):
    r = random.Random(seed)
    return f"[doc {seed}]\n" + " ".join(f"{r.choice(WORDS)}{r.randint(0, 999)}" for _ in range(int(n / 2.4))) + "\n[end doc]"


def stream_one(base, seed, n_ctx, gen_tokens, out, idx):
    msgs = [{"role": "user", "content": doc(seed, n_ctx) + "\n\nWrite a 300-word free-form analysis of the document."}]
    body = {"model": MODEL, "messages": msgs, "temperature": 0, "max_tokens": gen_tokens,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if first is None and chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                first = time.time()
            if chunk.get("usage"):
                usage = chunk["usage"]
    last = time.time()
    ct = usage.get("completion_tokens", 0)
    dec = ct / (last - first) if first and last > first and ct else 0.0
    out[idx] = {"ttft": (first - t0) if first else 0.0, "decode": dec, "ct": ct}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18000")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--ctx-tokens", type=int, default=8000)
    ap.add_argument("--gen-tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=int(time.time()))
    a = ap.parse_args()

    out = [None] * a.concurrency
    threads = [threading.Thread(target=stream_one, args=(a.base, a.seed * 100 + i, a.ctx_tokens, a.gen_tokens, out, i))
               for i in range(a.concurrency)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    decs = [o["decode"] for o in out if o]
    print(f"C{a.concurrency} ctx~{a.ctx_tokens}: per-stream decode " +
          " ".join(f"{d:.1f}" for d in decs) +
          f" | mean {sum(decs)/len(decs):.1f} | aggregate {sum(decs):.1f} tok/s | "
          f"ttft {min(o['ttft'] for o in out):.1f}-{max(o['ttft'] for o in out):.1f}s | wall {wall:.1f}s")


if __name__ == "__main__":
    main()
