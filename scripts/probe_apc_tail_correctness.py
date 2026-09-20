#!/usr/bin/env python3
"""Correctness gate for the `[glm53-apc-tail-floor]` overlay (task 45 site 1).

The overlay moves where a reusable Mamba/MLA tail is registered, from `n` to
`floor((n - 1) / unit) * unit`. That is a *cache-key* change, so the questions
are "does a hit serve state that matches the position its key proves" and "did
the reuse actually happen".

Why an answer check alone is not a gate. Prefix caching is documented as
"almost a free lunch" that "won't change model outputs", and it is **prefill
only**: a broken cache still returns the right answer, just slowly. A gate that
only checks the answer therefore passes on a run with **zero reuse** — it cannot
distinguish a working cache from an inert one. Every case below asserts the
**hit boundary** it expects, the **query accounting**, and a **successful cache
reset**; the answer is a second, independent check.

The shared prefix is the instrument, so it is built once and reused. A pad of
fixed length is generated with a fixed seed and prepended, so the total length
lands on the hash grid *and* every case shares the same leading bytes. Cases
that regenerate their padding are not testing a shared prefix at all.

Shapes, all on one ~20k-token document with the code at the very end:

  cold               the document + a question, cold cache. Expects **0** hits:
                     the negative control for the reuse assertions themselves.
  exact_replay       byte-identical resubmit. Expects the full reachable ceiling
                     `(n - 1) // unit * unit`. With a hash-grid length that is
                     `n - unit`; stock registered at `n`, out of reach.
  append_continuation  the same prompt plus 32 new tokens, which is the agentic
                     follow-up shape. The consumer's own ceiling is `n`, but the
                     producer registered its tail at `n - unit`, so the expected
                     boundary is `n - unit`. **This is the accepted 64-token cost
                     of the fix, asserted rather than assumed** — if both tail
                     positions are ever registered (task 45b), this case's
                     expectation becomes `n`. Its answer is deliberately **not
                     checked**: the extension necessarily lands after the
                     question, so the continuation is not a well-formed answer.
                     The boundary and query accounting still are, and those are
                     what this shape exists to measure.
  changed_needle     the same document with a *different* code at the end, so
                     nearly the whole prefix is shared. Expects the shared-prefix
                     boundary and, critically, **no stale leak**: the old code
                     must not appear in the answer. This is the negative control
                     for state mispositioning.

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
NEW_NEEDLE = "PETREL-2277"
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


def reset(base: str) -> tuple[bool, str]:
    """Reset the prefix cache. Returns (ok, status). Never raises."""
    try:
        _post(base, "/reset_prefix_cache", {})
        return True, "ok"
    except Exception as e:  # noqa: BLE001 - probe reports, never dies
        return False, f"err:{type(e).__name__}"


def tokenize(base: str, text: str) -> list[int]:
    return list(_post(base, "/tokenize", {"model": MODEL, "prompt": text})["tokens"])


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


def shared_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def expected_boundary(shared_tokens: int, prompt_tokens: int, unit: int) -> int:
    """Deepest hash-aligned position the engine may hand back for this request.

    Bounded from above by the finder's own cap, `(num_tokens - 1) // unit * unit`,
    and by how much of the prefix is actually identical to a cached one.
    """
    ceiling = (prompt_tokens - 1) // unit * unit
    return min(shared_tokens // unit * unit, ceiling)


def evaluate(cases: list[dict]) -> tuple[bool, list[str]]:
    """Pure verdict over case records. Returns (ok, failures).

    Deliberately independent of the network so the gate's own behaviour can be
    unit-tested: a run with no reuse, a stale leak, an inconsistent query delta
    or a failed reset must all FAIL.
    """
    failures: list[str] = []
    for c in cases:
        label = c["case"]
        if not c.get("reset_ok", True):
            failures.append(f"{label}: cache reset failed ({c.get('reset_status')})")
        if c["queried_tokens"] != c["prompt_tokens"]:
            failures.append(
                f"{label}: query delta {c['queried_tokens']} does not account for the "
                f"{c['prompt_tokens']}-token prompt"
            )
        if c["hits"] != c["expected_hits"]:
            failures.append(
                f"{label}: reused {c['hits']} tokens, expected exactly "
                f"{c['expected_hits']} (ceiling {c['ceiling_tokens']})"
            )
        if not c["answer_ok"] and c.get("answer_checked", True):
            failures.append(f"{label}: expected {c['expect']!r} in the answer")
        if c.get("stale_leak"):
            failures.append(
                f"{label}: STALE LEAK - the superseded {c['stale_probe']!r} appeared "
                "in the answer"
            )
    return (not failures), failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("GLM53_BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--unit", type=int, default=64)
    ap.add_argument("--doc-words", type=int, default=8000)
    ap.add_argument("--append-tokens", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = args.base.rstrip("/")
    unit = args.unit

    rnd = random.Random(90210)
    head = " ".join(f"{rnd.choice(WORDS)}{rnd.randint(0, 999)}" for _ in range(400))
    body = " ".join(f"{rnd.choice(WORDS)}{rnd.randint(0, 999)}" for _ in range(args.doc_words))
    question = (
        "\n\nQuestion: what is the access code stated at the end of the document? "
        "Answer with the code only."
    )

    head_ids = tokenize(base, head)
    body_ids = tokenize(base, body)
    q_ids = tokenize(base, question)
    lead = "\nThe access code is "
    lead_ids = tokenize(base, lead)
    tail_ids = tokenize(base, ".\n[end document]")

    def body_for(needle: str) -> list[int]:
        return head_ids + body_ids + lead_ids + tokenize(base, needle) + tail_ids

    # Fixed-length pad, generated once. It is part of every case's shared prefix,
    # so regenerating it per case (as an earlier revision did) would silently
    # break the shared prefix and turn the reuse assertions into no-ops.
    base_body = body_for(NEEDLE)
    pad_len = (-(len(base_body) + len(q_ids))) % unit
    pad_ids = tokenize(
        base, " ".join(f"{rnd.choice(WORDS)}{rnd.randint(0, 999)}" for _ in range(400))
    )[:pad_len]
    assert len(pad_ids) == pad_len, (pad_len, len(pad_ids))

    def build(needle: str) -> list[int]:
        return pad_ids + body_for(needle) + q_ids

    prompt_ids = build(NEEDLE)
    n = len(prompt_ids)
    assert n % unit == 0, (n, unit)
    ceiling = (n - 1) // unit * unit
    print(f"# prompt {n} tok (hash-grid aligned), pad {pad_len}, unit {unit}, "
          f"ceiling {ceiling}")

    extra_ids = tokenize(
        base, " ".join(f"{rnd.choice(WORDS)}{rnd.randint(0, 999)}" for _ in range(80))
    )[: args.append_tokens]
    assert len(extra_ids) == args.append_tokens, (args.append_tokens, len(extra_ids))
    append_ids = prompt_ids + extra_ids
    append_ceiling = (len(append_ids) - 1) // unit * unit

    changed_ids = build(NEW_NEEDLE)
    changed_shared = shared_prefix_len(prompt_ids, changed_ids)

    # (label, prompt, shared_tokens, reset_first, stale_probe, answer_checked)
    plan = [
        ("cold", prompt_ids, 0, True, None, True),
        ("exact_replay", prompt_ids, n, False, None, True),
        ("append_continuation", append_ids, n, False, None, False),
        ("changed_needle", changed_ids, changed_shared, False, NEEDLE, True),
    ]

    cases: list[dict] = []
    for label, ids, shared, do_reset, stale, answer_checked in plan:
        status = "not-requested"
        ok_reset = True
        if do_reset:
            ok_reset, status = reset(base)
            print(f"# reset before {label}: {status}")
        m0 = metrics(base)
        res = chat(base, ids)
        m1 = metrics(base)
        text = res["text"]
        ptok = int(res["usage"].get("prompt_tokens", len(ids)))
        expect = NEW_NEEDLE if label == "changed_needle" else NEEDLE
        # The consumer cannot reuse more of the prefix than it shares with a
        # cached request. For the append shape the producer's registered tail is
        # n - unit (the accepted cost), which is below the consumer's own ceiling.
        exp_hits = (
            n - unit if label == "append_continuation"
            else expected_boundary(shared, ptok, unit)
        )
        case = {
            "case": label,
            "prompt_tokens": ptok,
            "shared_prefix_tokens": shared,
            "hits": int(m1["hits"] - m0["hits"]),
            "queried_tokens": int(m1["queries"] - m0["queries"]),
            "ceiling_tokens": append_ceiling if label == "append_continuation" else ceiling,
            "expected_hits": exp_hits,
            "reset_ok": ok_reset,
            "reset_status": status,
            "expect": expect,
            "answer_ok": expect.lower() in text.lower(),
            "answer_checked": answer_checked,
            "stale_probe": stale,
            "stale_leak": bool(stale and stale.lower() in text.lower()),
            "wall_s": res["wall_s"],
            "text": text[:400],
        }
        cases.append(case)
        print(
            f"{label:>20}  prompt={ptok:>6}  shared={shared:>6}  "
            f"hits={case['hits']:>6}  expected={exp_hits:>6}  "
            f"queries={case['queried_tokens']:>6}  ok={case['answer_ok']}"
            f"{'' if answer_checked else ' (answer not checked: continuation)'}"
            f"{'  STALE!' if case['stale_leak'] else ''}  wall={case['wall_s']}s\n"
            f"{'':>20}  -> {text.strip()[:100]!r}"
        )

    ok, failures = evaluate(cases)
    payload = {
        "probe": "apc_tail_correctness",
        "schema": 2,
        "ts": datetime.now(timezone.utc).isoformat(),
        "base": base,
        "needle": NEEDLE,
        "new_needle": NEW_NEEDLE,
        "unit": unit,
        "pad_tokens": pad_len,
        "prompt_tokens": n,
        "hash_grid_aligned": True,
        "ceiling": ceiling,
        "append_tokens": args.append_tokens,
        "append_ceiling": append_ceiling,
        "accepted_append_cost_tokens": unit,
        "ok": ok,
        "failures": failures,
        "cases": cases,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"# wrote {args.out}")
    for f_ in failures:
        print(f"# FAIL {f_}")
    print(
        f"# VERDICT: {'PASS' if ok else 'FAIL'} "
        "(hit boundary, query accounting, reset and answer in every case; "
        "no stale leak)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
