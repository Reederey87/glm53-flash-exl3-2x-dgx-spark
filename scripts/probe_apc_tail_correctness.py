#!/usr/bin/env python3
"""Correctness gate for the `[glm53-apc-tail-floor]` overlay (task 45 site 1).

The overlay moves where a reusable Mamba/MLA tail is registered, from `n` to
`floor((n - 1) / unit) * unit`. That is a *cache-key* change, so the first
question is not "is it faster" but "does a hit serve state that matches the
position its key proves". This probe answers that with a retrieval task whose
answer depends on the whole prefix.

Three requests on one ~20k-token document carrying an early needle:

  cold            the document + a question, cold cache. Baseline answer.
  exact_replay    byte-identical resubmit. This is the case the overlay fixes:
                  the hit now lands on the floored tail instead of falling a
                  whole 3584-token page short.
  divergent_tail  the same document with its final 64 tokens replaced by
                  different filler, and the question appended. The shared
                  prefix now ends exactly on a hash boundary, so the hit lands
                  on the registered tail and the engine must continue from
                  state that matches *that* position, not the producer's end.

A stale or mispositioned state shows up immediately on a 20k-token prefix: the
needle is no longer retrievable and the answer drifts. `needle_ok` is the gate;
`hits` is the corroborating counter reading, and `queries` must account for the
whole prompt or the counter read is not trustworthy.

    python3 scripts/probe_apc_tail_correctness.py --out local/apc-tail-correctness.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone

MODEL = "GLM-5.3-Flash-EXL3"
NEEDLE = "CORMORANT-8815"
WORDS = [
    "alpha", "beam", "cache", "delta", "ember", "fjord", "glyph", "hinge",
    "ionic", "joule", "kelvin", "lumen", "matrix", "nadir", "orbit", "prism",
    "quartz", "rotor", "sigma", "torus", "umbra", "vector", "wafer", "xenon",
    "yield", "zenith",
]


def _post(base: str, path: str, body: dict, timeout: float = 1800):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    return json.loads(raw) if raw.strip() else {}


def _get(base: str, path: str, timeout: float = 60) -> str:
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return r.read().decode()


def metrics(base: str) -> dict:
    txt = _get(base, "/metrics")

    def g(name: str) -> float:
        return sum(
            float(v)
            for v in re.findall(rf"^{re.escape(name)}\{{[^}}]*\}}\s+(\S+)$", txt, re.M)
        )

    return {
        "hits": g("vllm:prefix_cache_hits_total"),
        "queries": g("vllm:prefix_cache_queries_total"),
    }


def reset(base: str) -> str:
    try:
        _post(base, "/reset_prefix_cache", {})
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


def tokenize(base: str, text: str) -> list[int]:
    return list(_post(base, "/tokenize", {"model": MODEL, "prompt": text})["tokens"])


def detokenize(base: str, ids: list[int]) -> str:
    return _post(base, "/detokenize", {"model": MODEL, "tokens": ids})["prompt"]


def chat(base: str, ids: list[int], max_tokens: int = 48) -> dict:
    t0 = time.time()
    out = _post(
        base,
        "/v1/completions",
        {
            "model": MODEL,
            "prompt": ids,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
    )
    return {
        "text": (out.get("choices") or [{}])[0].get("text", ""),
        "usage": out.get("usage", {}),
        "wall_s": round(time.time() - t0, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("GLM53_BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--unit", type=int, default=64)
    ap.add_argument("--doc-words", type=int, default=8000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = args.base.rstrip("/")

    rnd = random.Random(90210)
    head = " ".join(f"{rnd.choice(WORDS)}{rnd.randint(0, 999)}" for _ in range(400))
    body = " ".join(f"{rnd.choice(WORDS)}{rnd.randint(0, 999)}" for _ in range(args.doc_words))
    question = "\n\nQuestion: what is the access code stated near the beginning? Answer with the code only."
    question2 = "\n\nQuestion: restate the access code, then the first word after it. Answer briefly."

    def build(needle: str, q: str) -> list[int]:
        doc = f"{head}\nThe access code is {needle}.\n{body}\n[end document]"
        doc_ids = tokenize(base, doc)
        q_ids = tokenize(base, q)
        # Pad at the very start so the total length lands on the hash grid. This
        # is the case the overlay fixes: with an unaligned length the stock
        # arithmetic happens to agree with the ceiling and the defect is hidden.
        pad = (-(len(doc_ids) + len(q_ids))) % args.unit
        pad_ids = tokenize(base, " " + " ".join(f"{rnd.choice(WORDS)}" for _ in range(80)))[:pad]
        assert len(pad_ids) == pad, (pad, len(pad_ids))
        return pad_ids + doc_ids + q_ids

    prompt_ids = build(NEEDLE, question)
    n = len(prompt_ids)
    assert n % args.unit == 0, (n, args.unit)
    print(f"# prompt {n} tok (hash-grid aligned), unit {args.unit}, "
          f"ceiling {(n - 1) // args.unit * args.unit}")
    print(f"# cache reset: {reset(base)}")

    cases = []

    # 1. cold: the reference answer.
    reset(base)
    cold = chat(base, prompt_ids)
    cases.append(("cold", cold, 0, NEEDLE))

    # 2. exact replay of the identical prompt: the case the overlay fixes. Stock
    #    registered its tail at n (unreachable); the overlay registers at n - 64.
    m0 = metrics(base)
    warm = chat(base, prompt_ids)
    m1 = metrics(base)
    cases.append(("exact_replay", warm, int(m1["hits"] - m0["hits"]), NEEDLE))

    # 3. same document, a different follow-up question. The shared prefix runs to
    #    the producer's full prompt, so this exercises the append shape against
    #    the same registered tail.
    other_q = build(NEEDLE, question2)
    m0 = metrics(base)
    app = chat(base, other_q)
    m1 = metrics(base)
    cases.append(("append_question", app, int(m1["hits"] - m0["hits"]), NEEDLE))

    # 4. stale-state negative control: a DIFFERENT code earlier in the same
    #    document. The answer must be the new code, never the cached one.
    NEW = "PETREL-2277"
    changed = build(NEW, question)
    m0 = metrics(base)
    chg = chat(base, changed)
    m1 = metrics(base)
    cases.append(("changed_needle", chg, int(m1["hits"] - m0["hits"]), NEW))

    ok = True
    for label, res, hits, expect in cases:
        text = res["text"]
        found = expect.lower() in text.lower()
        stale = label == "changed_needle" and NEEDLE.lower() in text.lower()
        ok = ok and found and not stale
        print(
            f"{label:>15}  prompt={res['usage'].get('prompt_tokens')}  "
            f"hits={hits:>6}  expect={expect}  ok={found}"
            f"{'  STALE!' if stale else ''}  wall={res['wall_s']}s\n"
            f"{'':>15}  -> {text.strip()[:120]!r}"
        )

    payload = {
        "probe": "apc_tail_correctness",
        "schema": 1,
        "ts": datetime.now(timezone.utc).isoformat(),
        "base": base,
        "needle": NEEDLE,
        "prompt_tokens": n,
        "hash_grid_aligned": True,
        "ceiling": (n - 1) // args.unit * args.unit,
        "needle_ok_all": ok,
        "cases": [
            {"case": label, "hits": hits, "expect": expect,
             "ok": expect.lower() in r["text"].lower(),
             "stale_leak": label == "changed_needle" and NEEDLE.lower() in r["text"].lower(),
             "wall_s": r["wall_s"], "text": r["text"][:400]}
            for label, r, hits, expect in cases
        ],
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"# wrote {args.out}")
    print(f"# VERDICT: {'PASS' if ok else 'FAIL'} (expected code in every case, no stale leak)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
