#!/usr/bin/env python3
"""cache-burst.py — synthetic concurrent agentic-session load, from the Mac.

LOCAL to this cluster; not part of the upstream MiaAI-Lab kit.

Simulates N long-lived agent sessions (openclaw-shaped: big stable prompt
prefix, tiny new suffix per turn, short answers) hitting the server
concurrently for R rounds, and reports the server-side prefix-cache delta
per round. Round 1 is cold by construction; rounds 2+ measure whether each
session's cached prefix SURVIVED the other sessions' traffic — the eviction
churn probe. If N*ctx_tokens exceeds the 929,670-token KV pool, hit% on
rounds 2+ collapses; if it fits, hit% should be >90%.

Usage:
  uv run python local/cache-burst.py --sessions 4 --ctx-tokens 60000 --rounds 2
"""
import argparse, json, random, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

def metrics(base):
    with urllib.request.urlopen(f"{base}/metrics", timeout=15) as r:
        text = r.read().decode()
    def g(pat):
        m = re.search(pat, text, re.M)
        return float(m.group(1)) if m else 0.0
    return {
        "q": g(r'^vllm:prefix_cache_queries_total\{[^}]*\} (\S+)'),
        "h": g(r'^vllm:prefix_cache_hits_total\{[^}]*\} (\S+)'),
        "p": g(r'^vllm:prompt_tokens_total\{[^}]*\} (\S+)'),
    }

WORDS = ("ship hull engine cargo route port fuel speed wave tide deck crew "
         "chart wind storm anchor ballast draft keel bow stern radar sonar").split()

def make_prefix(seed, approx_tokens):
    rng = random.Random(seed)
    # ~1 token per short word for this tokenizer class; close enough for sizing
    return " ".join(rng.choice(WORDS) for _ in range(approx_tokens))

def turn(base, model, prefix, session, rnd, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": f"You are agent {session}. Context log follows."},
            {"role": "user", "content": f"{prefix}\n\nTurn {rnd}: reply with one short sentence about item {rnd}."},
        ],
        "temperature": 0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            json.load(r)
        return time.time() - t0, None
    except Exception as e:
        return time.time() - t0, str(e)[:200]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18000")
    ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--ctx-tokens", type=int, default=60000)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()

    prefixes = [make_prefix(1000 + i, a.ctx_tokens) for i in range(a.sessions)]
    total = a.sessions * a.ctx_tokens
    print(f"sessions={a.sessions} ctx~{a.ctx_tokens} tok each "
          f"(~{total:,} total vs 929,670 pool) rounds={a.rounds} conc={a.concurrency}")

    for rnd in range(1, a.rounds + 1):
        m0, t0 = metrics(a.base), time.time()
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            res = list(ex.map(lambda i: turn(a.base, a.model, prefixes[i], i, rnd,
                                             a.max_tokens, a.timeout),
                              range(a.sessions)))
        wall, m1 = time.time() - t0, metrics(a.base)
        dq, dh, dp = m1["q"] - m0["q"], m1["h"] - m0["h"], m1["p"] - m0["p"]
        errs = [e for _, e in res if e]
        lat = sorted(t for t, _ in res)
        print(f"round {rnd}: wall {wall:6.1f}s  hit% {100*dh/max(dq,1):5.1f}  "
              f"prompt_tok {dp:9.0f}  prefill_tps {dp/max(wall,0.01):7.0f}  "
              f"lat p50/max {lat[len(lat)//2]:.1f}/{lat[-1]:.1f}s  errors {len(errs)}")
        for e in errs[:3]:
            print(f"    err: {e}")

if __name__ == "__main__":
    main()
