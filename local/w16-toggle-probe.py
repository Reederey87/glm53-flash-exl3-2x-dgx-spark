#!/usr/bin/env python3
"""W16 probe: does a thinking on/off toggle keep the prefix cache?

Sends ONE unique ~N-token prompt through a fixed turn sequence and reads the
vllm:prefix_cache_{queries,hits}_total counter deltas per turn (the operative
ledger on this fork — usage.prompt_tokens_details is null). Before W16 the
`Reasoning Effort` head line was gated on thinking_enabled, so turns 3/4
(toggle off / on) diverged at token ~2 and re-prefilled the whole prompt.

  turn 1  thinking ON   cold        expect hits ~0
  turn 2  thinking ON   warm        expect hits ~ prompt (page-aligned pre-W18)
  turn 3  thinking OFF  toggle      W16 pass = hits ~ prompt (was ~0)
  turn 4  thinking ON   toggle back W16 pass = hits ~ prompt
  turn 5  thinking OFF  effort=low  documented fork: a DIFFERENT effort still misses

Run from the Mac through the tunnel:
  uv run python local/w16-toggle-probe.py --tokens 21000 [--base http://127.0.0.1:18000]
Also checks (--check-max-tokens) that a request omitting max_tokens ends with
finish_reason=stop and reports the server-side default (W16 G).
"""
import argparse, json, random, re, time, urllib.request

MODEL = "GLM-5.3-Flash-EXL3"


def metrics(base):
    with urllib.request.urlopen(f"{base}/metrics", timeout=15) as r:
        txt = r.read().decode()
    def g(pat):
        m = re.search(pat, txt, re.M)
        return float(m.group(1)) if m else 0.0
    return {"q": g(r'^vllm:prefix_cache_queries_total\{[^}]*\} (\S+)'),
            "h": g(r'^vllm:prefix_cache_hits_total\{[^}]*\} (\S+)')}


def make_doc(seed, approx_tokens):
    rnd = random.Random(seed)
    words = ["alpha", "beam", "cache", "delta", "ember", "fjord", "glyph", "hinge", "ionic",
             "joule", "kelvin", "lumen", "matrix", "nadir", "orbit", "prism", "quartz", "rotor",
             "sigma", "torus", "umbra", "vector", "wafer", "xenon", "yield", "zenith"]
    # ~1.3 tokens/word on this tokenizer for plain words with numbers mixed in
    n = int(approx_tokens / 1.3)
    body = " ".join(f"{rnd.choice(words)}{rnd.randint(0, 999)}" for _ in range(n))
    return f"[doc {seed}]\n{body}\n[end doc]"


def chat(base, messages, enable_thinking, effort=None, max_tokens=1, timeout=900):
    body = {"model": MODEL, "messages": messages, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": enable_thinking}}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if effort is not None:
        body["chat_template_kwargs"]["reasoning_effort"] = effort
    req = urllib.request.Request(f"{base}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    return time.time() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18000")
    ap.add_argument("--tokens", type=int, default=21000)
    ap.add_argument("--seed", type=int, default=int(time.time()))
    ap.add_argument("--check-max-tokens", action="store_true")
    a = ap.parse_args()

    doc = make_doc(a.seed, a.tokens)
    msgs = [{"role": "system", "content": "You are a careful assistant. Answer in one word."},
            {"role": "user", "content": doc + "\n\nWhat is the first word after '[doc'? One word."}]
    plan = [("1 thinking ON  cold", True, None),
            ("2 thinking ON  warm", True, None),
            ("3 thinking OFF toggle", False, None),
            ("4 thinking ON  toggle back", True, None),
            ("5 thinking OFF effort=low", False, "low")]
    rows = []
    for label, think, effort in plan:
        m0 = metrics(a.base)
        dt, out = chat(a.base, msgs, think, effort)
        m1 = metrics(a.base)
        q, h = m1["q"] - m0["q"], m1["h"] - m0["h"]
        pt = out.get("usage", {}).get("prompt_tokens", 0)
        rows.append((label, pt, int(q), int(h), (h / q * 100 if q else 0.0), dt))
        print(f"{label:28s} prompt={pt:6d} queries={int(q):6d} hits={int(h):6d} "
              f"hit%={(h / q * 100 if q else 0):6.1f} wall={dt:6.2f}s")
    t2, t3, t4 = rows[1][4], rows[2][4], rows[3][4]
    verdict = "PASS" if (t3 >= 95 and t4 >= 95) else "FAIL"
    print(f"\nW16 toggle verdict: {verdict} (warm {t2:.1f}% / off-toggle {t3:.1f}% / on-toggle {t4:.1f}%); "
          f"turn 5 (different effort) {rows[4][4]:.1f}% — expected to miss on this fork")
    if a.check_max_tokens:
        dt, out = chat(a.base, [{"role": "user", "content": "Reply with the single word: ready"}],
                       False, max_tokens=None)
        ch = out["choices"][0]
        print(f"\nno-max_tokens request: finish_reason={ch.get('finish_reason')} "
              f"completion_tokens={out.get('usage', {}).get('completion_tokens')} wall={dt:.2f}s")


if __name__ == "__main__":
    main()
