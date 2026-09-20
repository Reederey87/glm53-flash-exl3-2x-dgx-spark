#!/usr/bin/env python3
"""Probe: which prefix-cache boundaries are actually reachable on this hybrid.

Task 45 asks for the symptom to be *reproduced* before any fix is ported. This
probe measures the gap between the deepest position a replay can reach and the
deepest position the engine actually registered, on the live server.

Why a probe and not a code reading. The hybrid coordinator reconciles one hit
length across a fine-grained MLA group (alignment 64 here), four dense Mamba
groups (block 3584) and the DFlash2 drafter SWA group (block 64, EAGLE). The
reconciled hit is the `min()` of what each group can serve, so a group that
registers its tail one unit above any lookupable position silently caps the
whole request at the previous full page. Reading three interlocking formulas
does not tell you which one binds; measuring does.

The engine's own contract, read from the deployed tree
(`vllm/v1/core/kv_cache_manager.py`):

    max_cache_hit_length = request.num_tokens - 1

because the last token must be recomputed to obtain logits. So for a request of
``N`` prompt tokens the deepest *lookupable* hash unit is
``floor((N - 1) / unit) * unit`` with ``unit = hash_block_size`` (64 with
fine-grained APC, 3584 without). A producer that registers reusable state at a
position above that ceiling has registered it where no consumer can ask for it.

Shapes measured, because they differ and only one of them is the agentic one:

  exact      consumer prompt == producer prompt. Ceiling ``(N-1)//unit*unit``.
  append     consumer prompt == producer prompt + K new tokens (the agentic
             follow-up shape). Its own ceiling is ``(N+K-1)//unit*unit``, so a
             boundary registered at ``N`` *is* reachable here.
  page       producer prompt an exact multiple of the 3584-token hybrid page.

Both ``exact`` and ``append`` are needed: a defect that only bites ``exact``
would look like a cache that works, because agentic turns append.

Counters. ``vllm:prefix_cache_hits_total`` and ``vllm:prefix_cache_queries_total``
are monotonic counters, so each case is measured as a delta around one request,
with ``POST /reset_prefix_cache`` between cases. ``queries`` should account for
the whole prompt, which is the internal consistency check on the measurement.

Stdlib only, so it runs on the node with the system interpreter.

    python3 scripts/probe_apc_boundary_reachability.py --out local/apc-boundary.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

MODEL = "GLM-5.3-Flash-EXL3"
WORDS = [
    "alpha", "beam", "cache", "delta", "ember", "fjord", "glyph", "hinge",
    "ionic", "joule", "kelvin", "lumen", "matrix", "nadir", "orbit", "prism",
    "quartz", "rotor", "sigma", "torus", "umbra", "vector", "wafer", "xenon",
    "yield", "zenith",
]


def _post(base: str, path: str, body: dict | None, timeout: float = 1800):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "POST",
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
        # Sum across label sets: this deployment exports one series per engine.
        vals = re.findall(rf"^{re.escape(name)}\{{[^}}]*\}}\s+(\S+)$", txt, re.M)
        return sum(float(v) for v in vals)

    return {
        "hits": g("vllm:prefix_cache_hits_total"),
        "queries": g("vllm:prefix_cache_queries_total"),
    }


def reset_cache(base: str) -> str:
    """Reset the prefix cache. Returns a short status string, never raises."""
    for path in ("/reset_prefix_cache", "/flush_cache"):
        try:
            _post(base, path, {})
            return f"ok:{path}"
        except urllib.error.HTTPError as e:
            if e.code in (404, 405, 501):
                continue
            return f"err:{path}:{e.code}"
        except Exception as e:  # noqa: BLE001 - probe reports, never dies
            return f"err:{path}:{type(e).__name__}"
    return "unavailable"


def tokenize(base: str, text: str) -> list[int]:
    out = _post(base, "/tokenize", {"model": MODEL, "prompt": text})
    return list(out["tokens"])


def doc_tokens(base: str, seed: int, n_words: int) -> list[int]:
    rnd = random.Random(seed)
    text = " ".join(f"{rnd.choice(WORDS)}{rnd.randint(0, 999)}" for _ in range(n_words))
    return tokenize(base, text)


def complete(base: str, ids: list[int], max_tokens: int = 1) -> dict:
    """One raw completion on an exact token-id prompt. Returns usage + timing."""
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
    return {"usage": out.get("usage", {}), "wall_s": round(time.time() - t0, 3)}


def run_case(base: str, label: str, producer: list[int], consumer: list[int],
             unit: int, reset: bool) -> dict:
    """Cold-prime ``producer``, then measure what ``consumer`` reuses."""
    if reset:
        reset_cache(base)
    prime = complete(base, producer)
    n_prime = int(prime["usage"].get("prompt_tokens", len(producer)))

    m0 = metrics(base)
    t0 = time.time()
    warm = complete(base, consumer)
    wall = time.time() - t0
    m1 = metrics(base)

    n_cons = int(warm["usage"].get("prompt_tokens", len(consumer)))
    hits = m1["hits"] - m0["hits"]
    queries = m1["queries"] - m0["queries"]
    # Deepest position the engine is allowed to hand back for this consumer.
    ceiling = (n_cons - 1) // unit * unit
    shared = 0
    for a, b in zip(producer, consumer):
        if a != b:
            break
        shared += 1
    return {
        "case": label,
        "producer_tokens": n_prime,
        "consumer_tokens": n_cons,
        "shared_prefix_tokens": shared,
        "reused_tokens": int(hits),
        "queried_tokens": int(queries),
        "ceiling_tokens": ceiling,
        "reach_ratio": round(hits / ceiling, 4) if ceiling else None,
        "unreachable_tokens": int(ceiling - hits),
        "prime_wall_s": prime["wall_s"],
        "warm_wall_s": round(wall, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("GLM53_BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--unit", type=int, default=64,
                    help="hash unit: 64 with fine-grained APC, 3584 without")
    ap.add_argument("--page", type=int, default=3584)
    ap.add_argument("--ladder", default="7360,10752,14336,6464,6500,10000,20000,7168,10752")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-reset", action="store_true",
                    help="do not reset between cases (measures retention instead)")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    try:
        health = _get(base, "/health").strip()
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: {base}/health unreachable: {e}", file=sys.stderr)
        return 2
    if health != "200":
        # /health answers 200 with an empty body on some builds; accept anything.
        pass

    print(f"# apc boundary reachability probe — {base} — unit={args.unit} page={args.page}")
    print(f"# cache reset: {reset_cache(base)}")

    # One long token pool, sliced to exact lengths. Slicing keeps every case on
    # the same content, so a hit difference is a boundary difference.
    pool = doc_tokens(base, 4242, 40000)
    if len(pool) < 21000:
        print(f"FATAL: tokenizer produced only {len(pool)} tokens", file=sys.stderr)
        return 2

    results = []
    for spec in args.ladder.split(","):
        spec = spec.strip()
        if not spec:
            continue
        n = int(spec)
        if n >= len(pool) - 64:
            print(f"# skip N={n}: pool too short")
            continue
        producer = pool[:n]
        # exact replay: same prompt. Its own ceiling is (n-1)//unit*unit, so a
        # boundary registered at n is out of reach by construction.
        results.append(run_case(base, f"exact@{n}", producer, producer,
                                args.unit, not args.no_reset))
        # agentic follow-up: the same history plus new tokens. Ceiling rises
        # above n, so a boundary registered at n is reachable here.
        follow = pool[: n + 32]
        results.append(run_case(base, f"append32@{n}", producer, follow,
                                args.unit, not args.no_reset))
        # page-aligned producer: _cache_partial_tail_block refuses these
        # (num_tokens % block_size == 0), which is the second suspect.
        if n % args.page == 0:
            results.append(run_case(base, f"page_aligned@{n}", producer, producer,
                                    args.unit, not args.no_reset))

    for r in results:
        print(
            f"{r['case']:>20}  prompt={r['consumer_tokens']:>6}  "
            f"shared={r['shared_prefix_tokens']:>6}  "
            f"reused={r['reused_tokens']:>6}  ceiling={r['ceiling_tokens']:>6}  "
            f"ratio={r['reach_ratio']}  short={r['unreachable_tokens']:>6}  "
            f"queries={r['queried_tokens']:>6}  warm={r['warm_wall_s']}s"
        )

    payload = {
        "probe": "apc_boundary_reachability",
        "schema": 1,
        "ts": datetime.now(timezone.utc).isoformat(),
        "base": base,
        "unit": args.unit,
        "page": args.page,
        "reset_between_cases": not args.no_reset,
        "results": results,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"# wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
