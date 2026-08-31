#!/usr/bin/env python3
"""W18 probe: prefix-cache hit granularity + cold-vs-replay output stability.

Part A — follow-up ladder. For each prompt size N, send a unique document twice:
turn 1 cold, turn 2 = the same messages + a ~200-token appended user turn (the
agentic follow-up shape). Reads vllm:prefix_cache_{queries,hits}_total deltas.
Pre-W18 (page-aligned hits) turn 2 can reuse at most floor(N/3584)*3584 tokens
— and NOTHING for N < 3584. Post-W18 (hash grain 64) it should reuse ~N.

Part B — output stability. Three unique prompts, temp 0, thinking off,
max_tokens 48: cold completion vs immediate replay. Reports whether the two
completions are byte-identical (a hybrid KDA model may legitimately differ on
cached-vs-recomputed state; the control run on the pre-W18 boot is the reference).

  uv run python local/w18-fgapc-probe.py [--base http://127.0.0.1:18000]
"""
import argparse, json, random, re, time, urllib.request

MODEL = "GLM-5.3-Flash-EXL3"
PAGE = 3584


def metrics(base):
    with urllib.request.urlopen(f"{base}/metrics", timeout=15) as r:
        txt = r.read().decode()
    def g(pat):
        m = re.search(pat, txt, re.M)
        return float(m.group(1)) if m else 0.0
    return {"q": g(r'^vllm:prefix_cache_queries_total\{[^}]*\} (\S+)'),
            "h": g(r'^vllm:prefix_cache_hits_total\{[^}]*\} (\S+)')}


WORDS = ["alpha", "beam", "cache", "delta", "ember", "fjord", "glyph", "hinge", "ionic",
         "joule", "kelvin", "lumen", "matrix", "nadir", "orbit", "prism", "quartz", "rotor",
         "sigma", "torus", "umbra", "vector", "wafer", "xenon", "yield", "zenith"]


def doc(seed, approx_tokens):
    rnd = random.Random(seed)
    n = max(8, int(approx_tokens / 2.4))  # measured ~2.4 tok per "word123 " unit
    return f"[doc {seed}]\n" + " ".join(f"{rnd.choice(WORDS)}{rnd.randint(0, 999)}" for _ in range(n)) + "\n[end doc]"


def chat(base, messages, max_tokens=1, timeout=900):
    body = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    return time.time() - t0, out


def timed_turn(base, messages, max_tokens=1):
    m0 = metrics(base); dt, out = chat(base, messages, max_tokens); m1 = metrics(base)
    return dt, out, int(m1["q"] - m0["q"]), int(m1["h"] - m0["h"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18000")
    ap.add_argument("--seed", type=int, default=int(time.time()))
    a = ap.parse_args()

    print("== Part A: follow-up ladder (turn 2 = same prompt + ~200-token follow-up)")
    print(f"{'N~':>7} {'prompt1':>8} {'prompt2':>8} {'hits2':>7} {'page-floor':>10} {'hit%':>6} {'wall1':>7} {'wall2':>7}")
    for i, n in enumerate((2000, 5000, 8000, 30000)):
        d = doc(a.seed * 10 + i, n)
        m1 = [{"role": "user", "content": d + "\n\nReply with one word: ok."}]
        w1, o1, q1, h1 = timed_turn(a.base, m1)
        follow = " ".join(f"{WORDS[(i * 7 + k) % len(WORDS)]}{k}" for k in range(85))
        m2 = m1 + [{"role": "assistant", "content": "ok"},
                   {"role": "user", "content": follow + "\n\nReply with one word: done."}]
        w2, o2, q2, h2 = timed_turn(a.base, m2)
        p1 = o1["usage"]["prompt_tokens"]; p2 = o2["usage"]["prompt_tokens"]
        floor = (p1 // PAGE) * PAGE
        print(f"{n:>7} {p1:>8} {p2:>8} {h2:>7} {floor:>10} {(h2 / q2 * 100 if q2 else 0):>6.1f} {w1:>7.2f} {w2:>7.2f}")

    print("\n== Part B: temp-0 cold vs replay, 3 unique prompts, max_tokens 48")
    same = 0
    for i in range(3):
        d = doc(a.seed * 100 + i, 6000)
        m = [{"role": "user", "content": d + "\n\nSummarize the document in one sentence."}]
        _, o1, _, _ = timed_turn(a.base, m, 48)
        _, o2, _, h2 = timed_turn(a.base, m, 48)
        c1 = o1["choices"][0]["message"]["content"]; c2 = o2["choices"][0]["message"]["content"]
        ident = c1 == c2; same += ident
        print(f"prompt {i}: replay hits={h2} identical={ident}" + ("" if ident else f"\n   cold:   {c1[:120]!r}\n   replay: {c2[:120]!r}"))
    print(f"\nPart B: {same}/3 byte-identical")


if __name__ == "__main__":
    main()
