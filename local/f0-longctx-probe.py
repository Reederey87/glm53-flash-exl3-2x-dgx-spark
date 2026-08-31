#!/usr/bin/env python3
"""F0: long-context decode + DFlash2 acceptance ladder (kit issue #73 shape).

For each context size, build a unique ~N-token document, warm it once
(max_tokens=1, so the follow-up generation hits the prefix cache and the
measurement is decode-dominated), then run one generation per phase:

  prose      "Continue the analysis..." (free-form, 300 tokens, temp 0)
  structured "Count from 1 to 120, comma-separated." (300 tokens, temp 0)

Reports decode tok/s (completion_tokens / (wall - TTFT), TTFT from SSE first
token) and the spec-decode acceptance over the request from the
vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total deltas, plus
per-position acceptance from vllm:spec_decode_num_accepted_tokens_per_pos.

  uv run python local/f0-longctx-probe.py [--ctx 0 50000 150000] [--base ...]
"""
import argparse, json, random, re, time, urllib.request

MODEL = "GLM-5.3-Flash-EXL3"
WORDS = ["alpha", "beam", "cache", "delta", "ember", "fjord", "glyph", "hinge", "ionic",
         "joule", "kelvin", "lumen", "matrix", "nadir", "orbit", "prism", "quartz", "rotor",
         "sigma", "torus", "umbra", "vector", "wafer", "xenon", "yield", "zenith"]


def spec_metrics(base):
    with urllib.request.urlopen(f"{base}/metrics", timeout=15) as r:
        txt = r.read().decode()
    def g(pat):
        m = re.search(pat, txt, re.M)
        return float(m.group(1)) if m else 0.0
    pos = {}
    for m in re.finditer(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\} (\S+)', txt, re.M):
        pos[int(m.group(1))] = float(m.group(2))
    return {"drafts": g(r'^vllm:spec_decode_num_drafts_total\{[^}]*\} (\S+)'),
            "draft_tok": g(r'^vllm:spec_decode_num_draft_tokens_total\{[^}]*\} (\S+)'),
            "acc": g(r'^vllm:spec_decode_num_accepted_tokens_total\{[^}]*\} (\S+)'),
            "pos": pos}


def doc(seed, n):
    r = random.Random(seed)
    return f"[doc {seed}]\n" + " ".join(f"{r.choice(WORDS)}{r.randint(0, 999)}" for _ in range(int(n / 2.4))) + "\n[end doc]"


def stream_chat(base, messages, max_tokens, timeout=1800):
    body = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if ttft is None and chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                ttft = time.time() - t0
            if chunk.get("usage"):
                usage = chunk["usage"]
    return (ttft or 0.0), time.time() - t0, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18000")
    ap.add_argument("--ctx", type=int, nargs="+", default=[0, 50000, 150000])
    ap.add_argument("--gen-tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=int(time.time()))
    a = ap.parse_args()

    print(f"{'ctx~':>7} {'phase':<10} {'prompt':>8} {'decode tok/s':>12} {'accept':>7} {'per-step':>8}  per-position")
    for i, n in enumerate(a.ctx):
        d = doc(a.seed * 10 + i, n) if n else "You are analysing token statistics."
        tasks = [("prose", "Continue with a 250-word free-form analysis of patterns you notice in the document."),
                 ("structured", "Count from 1 to 120 as a comma-separated list. Output only the numbers.")]
        # warm the shared prefix once so generation measures decode, not prefill
        stream_chat(a.base, [{"role": "user", "content": d + "\n\nReply with one word: ok."}], 1)
        for phase, task in tasks:
            msgs = [{"role": "user", "content": d + "\n\n" + task}]
            s0 = spec_metrics(a.base)
            ttft, wall, usage = stream_chat(a.base, msgs, a.gen_tokens)
            s1 = spec_metrics(a.base)
            ct = usage.get("completion_tokens", 0); pt = usage.get("prompt_tokens", 0)
            dec = ct / (wall - ttft) if wall > ttft and ct else 0.0
            drafts = s1["drafts"] - s0["drafts"]; dtok = s1["draft_tok"] - s0["draft_tok"]
            acc = s1["acc"] - s0["acc"]
            ratio = acc / dtok if dtok else 0.0
            per_step = acc / drafts if drafts else 0.0
            pos = " ".join(f"{(s1['pos'].get(k,0)-s0['pos'].get(k,0))/drafts:.2f}" if drafts else "-" for k in sorted(set(s0["pos"]) | set(s1["pos"])))
            print(f"{n:>7} {phase:<10} {pt:>8} {dec:>12.1f} {ratio:>7.3f} {per_step:>8.2f}  {pos}")


if __name__ == "__main__":
    main()
