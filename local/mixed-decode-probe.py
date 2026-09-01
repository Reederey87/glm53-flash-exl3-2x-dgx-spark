#!/usr/bin/env python3
"""mixed-decode-probe.py — measure decode tok/s while a cold prefill co-runs (issue #6 shape).

LOCAL to this cluster; not part of the vendored upstream kit.

Starts a streaming decode (A), lets it reach steady state, then launches a cold
~pad-words prefill (B) and reports A's decode rate before / during / after B's
prefill window. On a build without the GLM53_MIXED_PREFILL_CHUNK=skip scheduler
patch the "during" rate collapses (upstream measured 55 -> 5 tok/s); with skip
active it should hold near the solo rate.

Usage: uv run python local/mixed-decode-probe.py
"""
import argparse, json, random, threading, time, urllib.request

def make_pad(seed, words):
    rng = random.Random(seed)
    w = "ship hull engine cargo route port fuel speed wave tide deck crew".split()
    return " ".join(rng.choice(w) for _ in range(words))

def decode_stream(base, model, events, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content":
                      "Write a very long, detailed story about a cargo ship's voyage. "
                      "Keep going until stopped."}],
        "temperature": 1.0, "max_tokens": max_tokens, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(time.time())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18000")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--pad-words", type=int, default=60000)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--warm-secs", type=int, default=15)
    a = ap.parse_args()

    events = []
    t = threading.Thread(target=decode_stream,
                         args=(a.base, a.model, events, a.max_tokens, 900), daemon=True)
    t.start()
    time.sleep(a.warm_secs)
    n_before = len(events)
    if n_before < 5:
        print("decode did not reach steady state; aborting")
        return
    rate_before = n_before / (events[-1] - events[0]) if n_before > 1 else 0

    # launch the cold prefill peer
    pad = make_pad(int(time.time()), a.pad_words)
    body = json.dumps({"model": a.model,
                       "messages": [{"role": "user", "content": pad + " Reply with one word."}],
                       "temperature": 0, "max_tokens": 8,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"{a.base}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t_b0 = time.time()
    def peer():
        with urllib.request.urlopen(req, timeout=900) as r:
            json.load(r)
    tb = threading.Thread(target=peer, daemon=True)
    tb.start()
    tb.join()
    t_b1 = time.time()

    during = [e for e in events if t_b0 <= e <= t_b1]
    rate_during = len(during) / (t_b1 - t_b0)
    t.join(timeout=120)
    after = [e for e in events if e > t_b1]
    rate_after = (len(after) / (after[-1] - after[0])) if len(after) > 1 else float("nan")

    # W26: the decode-side tail during the peer's crawl — inter-token gaps p50/p99 — is the
    # number that argues for a small late cap; the peer's wall time and A's tokens in a fixed
    # window after the peer starts are the numbers that argue for a larger one.
    gaps = sorted(b - a_ for a_, b in zip(during, during[1:]))
    def _pct(q):
        return gaps[min(len(gaps) - 1, int(q * len(gaps)))] if gaps else float("nan")
    fixed_win = t_b0 + 240.0
    in_win = sum(1 for e in events if t_b0 <= e <= fixed_win)
    print(f"peer cold prefill: ~{a.pad_words} words in {t_b1-t_b0:.1f}s")
    print(f"decode tok/s  before: {rate_before:5.1f}   DURING peer prefill: {rate_during:5.1f}"
          f"   after: {rate_after:5.1f}")
    print(f"collapse factor during/before: {rate_during/max(rate_before,0.01):.2f}x")
    print(f"decode gaps DURING peer: p50 {_pct(0.5):.2f}s  p99 {_pct(0.99):.2f}s  max {(gaps[-1] if gaps else float('nan')):.2f}s"
          f"   | decode tokens in the 240 s after peer start: {in_win}")

if __name__ == "__main__":
    main()
