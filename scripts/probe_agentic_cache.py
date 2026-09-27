#!/usr/bin/env python3
"""Gate battery for per-request prefix-cache attribution and agentic retention.

Two pre-registered gates, one instrument. Both are stated in `spec/TODO.md`
before the run and are read back here as explicit thresholds, so a receipt
cannot be re-interpreted after the numbers land.

  --mode attribution   per-request cache accounting is correct and readable.
  --mode retention     a known one-shot flood stops evicting a live session.
  --mode boundary      delegate to the standing APC boundary probe.
  --mode all           attribution, then both retention arms, then boundary.

Attribution gate (TODO item 1). The live argv used to omit
`--enable-prompt-tokens-details`, so `usage.prompt_tokens_details` was absent
and per-request reuse was unobservable; the kit's own cache probes had to fall
back to wall time and global counters. `GLM53_PROMPT_TOKENS_DETAILS=1` adds
upstream's flag. This mode proves the field is real rather than merely present:

  cold        a unique prompt reports `cached_tokens == 0` explicitly
  replay      the identical prompt reports `cached_tokens > 0`
  negative    a prompt whose prefix diverges early reports `cached_tokens == 0`
  sse         the streaming final usage chunk agrees with the HTTP replay value
  counters    for each isolated case `cached_tokens` equals that request's own
              `vllm:prefix_cache_hits_total` delta
  overload    six concurrent requests (above `MAX_NUM_SEQS=4`) record waiting,
              capacity waiting, preemptions, per-request TTFT, decode gaps and
              completed tokens, with no request error

Why the counter equality is the check and not a proxy. In the deployed tree
`KVCacheManager.record_prefix_cache_stats(request, num_new_local_computed_tokens)`
is the only writer of `prefix_cache_stats.hits`, and
`PrefillStats.set(num_local_cached_tokens=...)` receives the same
`num_new_local_computed_tokens`. One request therefore moves both numbers by
the same amount, and any difference means one of the two paths is not the one
believed. Global query counters are recorded, not asserted: under contention a
queued request can be looked up more than once, so their sum is not submitted
tokens.

Retention gate (TODO item 2). The write-side opt-out is already adopted
(`GLM53_APC_NO_STORE`, PR #18) and a request that carries
`vllm_xargs: {"skip_writing_prefix_cache": 1}` never inserts its blocks, so its
blocks stay unhashed and `BlockPool.free_blocks` recycles them from the front
of the free queue instead of queueing them behind a live session. That is a
per-request property, so both arms run in the SAME boot with the same server
argv: the chaff's no-store bit is the only variable, and no restart can
contaminate the comparison. Each arm gets its own `POST /reset_prefix_cache`
and a fresh salt, and the arm order is fixed by the pre-registration.

Why hot-protect does not already cover this. `GLM53_CACHE_HOT_PROTECT` marks a
block only on a request that *hits* it, so a session that has been written once
and not yet replayed is still one-shot pages when the flood arrives; the kit's
own receipt for the tool-return-grace arm measured 35,840/92,955 = 38.56% on
exactly that shape. This gate measures the same shape with the chaff opting out
of the write side.

Stdlib only, so it runs on the node with the system interpreter:

    python3 scripts/probe_agentic_cache.py --mode all --out local/agentic-cache-<date>

Nothing here changes server configuration. The instrument asserts the standing
controls it must not move and records them; it never writes them.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODEL = "GLM-5.3-Flash-EXL3"

WORDS = [
    "alpha", "beam", "cache", "delta", "ember", "fjord", "glyph", "hinge",
    "ionic", "joule", "kelvin", "lumen", "matrix", "nadir", "orbit", "prism",
    "quartz", "rotor", "sigma", "torus", "umbra", "vector", "wafer", "xenon",
    "yield", "zenith", "anvil", "basalt", "cinder", "dune", "estuary", "flint",
]

# Approximate tokens per salad word for this tokenizer. The probe never relies
# on it for a gate: every length it reports is the engine's own `prompt_tokens`.
TOKENS_PER_WORD = 1.25

# Pre-registered retention thresholds (TODO item 2).
RETENTION_MIN_GAIN_PP = 15.0
CHAFF_THROUGHPUT_FLOOR = 0.95

CODE_RE = re.compile(r"CODE-([0-9a-f]{12})")
NAN_RE = re.compile(r"\bnan\b|locklock", re.I)

METRIC_RE = re.compile(r"^(vllm:[A-Za-z0-9_:]+)(?:\{([^}]*)\})?\s+(\S+)$")


# ------------------------------------------------------------------ transport


class Client:
    def __init__(self, base: str, api_key: str = "", timeout: float = 1800.0):
        self.base = base.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _req(self, path: str, body: dict | None, timeout: float | None = None):
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"} if data else {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return urllib.request.Request(
            self.base + path, data=data, headers=headers,
            method="POST" if data is not None else "GET",
        )

    def get_text(self, path: str, timeout: float = 60.0) -> str:
        with urllib.request.urlopen(self._req(path, None), timeout=timeout) as r:
            return r.read().decode()

    def post_json(self, path: str, body: dict, timeout: float | None = None) -> dict:
        with urllib.request.urlopen(
            self._req(path, body, timeout), timeout=timeout or self.timeout
        ) as r:
            raw = r.read().decode()
        return json.loads(raw) if raw.strip() else {}

    def chat(self, body: dict, timeout: float | None = None) -> dict:
        return self.post_json("/v1/chat/completions", body, timeout)

    def chat_stream(self, body: dict, timeout: float | None = None) -> dict:
        """Stream one completion; return text, usage, TTFT and inter-token gaps."""
        req = self._req("/v1/chat/completions", body, timeout)
        text_parts: list[str] = []
        usage = None
        ttft = None
        gaps: list[float] = []
        last = None
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                now = time.monotonic()
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if not piece and not reasoning:
                        continue
                    if ttft is None:
                        ttft = now - t0
                    elif last is not None:
                        gaps.append(now - last)
                    last = now
                    text_parts.append(piece or reasoning)
        return {
            "text": "".join(text_parts),
            "usage": usage,
            "ttft_s": ttft,
            "gaps_s": gaps,
            "wall_s": time.monotonic() - t0,
        }

    def metrics(self) -> dict[str, float]:
        """Sum every vllm metric over its label sets; keep the raw lines too."""
        out: dict[str, float] = {}
        for line in self.get_text("/metrics").splitlines():
            if line.startswith("#"):
                continue
            m = METRIC_RE.match(line)
            if not m:
                continue
            name, _labels, value = m.group(1), m.group(2), m.group(3)
            try:
                v = float(value)
            except ValueError:
                continue
            if name.endswith("_created"):
                continue
            out[name] = out.get(name, 0.0) + v
        return out

    def metrics_labelled(self) -> dict[str, float]:
        """Same, keyed `name{labels}` so one label value can be read alone."""
        out: dict[str, float] = {}
        for line in self.get_text("/metrics").splitlines():
            if line.startswith("#"):
                continue
            m = METRIC_RE.match(line)
            if not m:
                continue
            name, labels, value = m.group(1), m.group(2) or "", m.group(3)
            try:
                v = float(value)
            except ValueError:
                continue
            if name.endswith("_created"):
                continue
            out[f"{name}{{{labels}}}"] = out.get(f"{name}{{{labels}}}", 0.0) + v
        return out

    def reset_prefix_cache(self) -> dict:
        return self.post_json("/reset_prefix_cache", {}, timeout=120.0)


def metric_snapshot(client: Client) -> dict[str, float]:
    return client.metrics()


def metric_delta(before: dict[str, float], after: dict[str, float],
                 name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


# ------------------------------------------------------------------- prompts


def salad(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(words))


def make_code(rng: random.Random) -> str:
    return "CODE-" + "".join(rng.choice("0123456789abcdef") for _ in range(12))


def build_prompt(rng: random.Random, target_tokens: int, code: str,
                 needle_depth: int, salt: str) -> str:
    """Long word-salad prompt with a unique access code buried in the prefix.

    The code sits `needle_depth` tokens in, i.e. inside the region a prefix-cache
    hit would serve, so a stale or wrong hit changes the answer. The salt makes
    every run unique even if a previous run's cache survived a reset.
    """
    total = max(int(target_tokens / TOKENS_PER_WORD), 64)
    head = max(int(needle_depth / TOKENS_PER_WORD), 8)
    return (
        f"Run identifier {salt}. Read the record below carefully.\n"
        f"{salad(rng, head)}\n"
        f"The access code for this record is {code}. Remember it.\n"
        f"{salad(rng, max(total - head, 16))}\n"
        f"Reply with only the access code, in the form CODE-xxxxxxxxxxxx, "
        f"and nothing else."
    )


def body_for(prompt: str, max_tokens: int, no_store: bool | None = None,
             stream: bool = False) -> dict:
    body: dict = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    if no_store is not None:
        body["vllm_xargs"] = {"skip_writing_prefix_cache": 1 if no_store else 0}
    return body


def prompt_tokens(usage: dict | None) -> int | None:
    return None if not usage else usage.get("prompt_tokens")


def cached_tokens(usage: dict | None) -> int | None:
    if not usage:
        return None
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return None
    return details.get("cached_tokens")


def extract_code(text: str) -> str | None:
    m = CODE_RE.search(text or "")
    return f"CODE-{m.group(1)}" if m else None


def contains_nan(text: str) -> bool:
    return bool(NAN_RE.search(text or ""))


# ------------------------------------------------------------------- cases


def one_request(client: Client, prompt: str, max_tokens: int,
                no_store: bool | None = None, stream: bool = False,
                label: str = "") -> dict:
    """One isolated request with its own engine-counter window around it."""
    body = body_for(prompt, max_tokens, no_store, stream)
    before = metric_snapshot(client)
    err = None
    try:
        if stream:
            res = client.chat_stream(body)
            usage = res["usage"]
            text = res["text"]
        else:
            out = client.chat(body)
            usage = out.get("usage")
            text = "".join(
                (c.get("message") or {}).get("content") or ""
                for c in out.get("choices") or []
            )
            res = {"ttft_s": None, "gaps_s": [], "wall_s": None}
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        err = f"{type(exc).__name__}: {exc}"
        usage, text = None, ""
        res = {"ttft_s": None, "gaps_s": [], "wall_s": None}
    after = metric_snapshot(client)
    return {
        "label": label,
        "stream": stream,
        "no_store": no_store,
        "prompt_tokens": prompt_tokens(usage),
        "cached_tokens": cached_tokens(usage),
        "usage_present": bool(usage),
        "details_present": bool(usage and usage.get("prompt_tokens_details") is not None),
        "completion_tokens": None if not usage else usage.get("completion_tokens"),
        "answer_code": extract_code(text),
        "nan": contains_nan(text),
        "error": err,
        "ttft_s": res["ttft_s"],
        "wall_s": res["wall_s"],
        "gaps_s": res["gaps_s"],
        "hits_delta": metric_delta(before, after, "vllm:prefix_cache_hits_total"),
        "queries_delta": metric_delta(before, after, "vllm:prefix_cache_queries_total"),
    }


def case_attribution(client: Client, args, rng: random.Random) -> dict:
    cases: dict[str, dict] = {}
    salt = args.salt

    client.reset_prefix_cache()
    prompt = build_prompt(rng, args.attr_tokens, make_code(rng), args.needle_depth, salt)
    cases["cold"] = one_request(client, prompt, args.attr_max_tokens, label="cold")
    cases["replay"] = one_request(client, prompt, args.attr_max_tokens, label="replay")

    # Changed-prefix negative control: same length, different early tokens.
    other = build_prompt(rng, args.attr_tokens, make_code(rng), args.needle_depth,
                         salt + "-neg")
    cases["negative"] = one_request(client, other, args.attr_max_tokens, label="negative")

    # The streaming path must agree with the non-streaming one on the same prompt.
    cases["replay_sse"] = one_request(client, prompt, args.attr_max_tokens,
                                      stream=True, label="replay_sse")

    # Above the four-slot admission limit: this is the concurrency observation,
    # not a speed claim.
    overload = run_overload(client, args, rng)
    return {"cases": cases, "overload": overload}


def run_overload(client: Client, args, rng: random.Random) -> dict:
    n = args.overload_requests
    prompts = [build_prompt(rng, args.attr_tokens, make_code(rng), args.needle_depth,
                            f"{args.salt}-c{i}") for i in range(n)]
    client.reset_prefix_cache()

    results: list[dict | None] = [None] * n
    errors: list[str] = []
    max_waiting = 0.0
    max_capacity_waiting = 0.0
    max_running = 0.0
    max_kv = 0.0
    stop = threading.Event()

    def sampler() -> None:
        nonlocal max_waiting, max_capacity_waiting, max_running, max_kv
        while not stop.is_set():
            try:
                m = client.metrics_labelled()
            except OSError:
                time.sleep(0.5)
                continue
            total_waiting = sum(v for k, v in m.items()
                                if k.startswith("vllm:num_requests_waiting{"))
            capacity = sum(v for k, v in m.items()
                           if k.startswith("vllm:num_requests_waiting_by_reason{")
                           and 'reason="capacity"' in k)
            running = sum(v for k, v in m.items()
                          if k.startswith("vllm:num_requests_running{"))
            kv = sum(v for k, v in m.items()
                     if k.startswith("vllm:kv_cache_usage_perc{"))
            max_waiting = max(max_waiting, total_waiting)
            max_capacity_waiting = max(max_capacity_waiting, capacity)
            max_running = max(max_running, running)
            max_kv = max(max_kv, kv)
            time.sleep(0.25)

    before = metric_snapshot(client)
    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    def worker(idx: int) -> None:
        body = body_for(prompts[idx], args.attr_max_tokens)
        t0 = time.monotonic()
        try:
            out = client.chat(body)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"request {idx}: {type(exc).__name__}: {exc}")
            return
        usage = out.get("usage")
        text = "".join(
            (c.get("message") or {}).get("content") or "" for c in out.get("choices") or []
        )
        results[idx] = {
            "prompt_tokens": prompt_tokens(usage),
            "cached_tokens": cached_tokens(usage),
            "completion_tokens": None if not usage else usage.get("completion_tokens"),
            "answer_code": extract_code(text),
            "expected_code": extract_code(prompts[idx]),
            "wall_s": time.monotonic() - t0,
        }

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    sampler_thread.join(timeout=5)
    after = metric_snapshot(client)

    return {
        "requests": n,
        "completed": sum(1 for r in results if r),
        "errors": errors,
        "max_waiting": max_waiting,
        "max_waiting_capacity": max_capacity_waiting,
        "max_running": max_running,
        "max_kv_cache_usage_perc": max_kv,
        "preemptions_delta": metric_delta(before, after, "vllm:num_preemptions_total"),
        "results": results,
    }


def retention_arm(client: Client, args, rng: random.Random, no_store: bool) -> dict:
    """One arm: two long sessions, a chaff flood, then the sessions again."""
    arm = "treatment" if no_store else "control"
    client.reset_prefix_cache()
    salt = f"{args.salt}-{arm}"

    agents = []
    for i in range(args.agent_count):
        code = make_code(rng)
        prompt = build_prompt(rng, args.agent_tokens, code, args.needle_depth,
                              f"{salt}-a{i}")
        res = one_request(client, prompt, args.agent_max_tokens, label=f"agent{i}")
        res["expected_code"] = code
        res["prompt"] = prompt
        agents.append(res)

    chaff = []
    chaff_t0 = time.monotonic()
    for i in range(args.chaff_count):
        code = make_code(rng)
        prompt = build_prompt(rng, args.chaff_tokens, code, args.needle_depth,
                              f"{salt}-x{i}")
        res = one_request(client, prompt, args.chaff_max_tokens,
                          no_store=no_store, label=f"chaff{i}")
        res["expected_code"] = code
        res.pop("prompt", None)
        chaff.append(res)
    chaff_wall = time.monotonic() - chaff_t0

    replays = []
    for i, agent in enumerate(agents):
        res = one_request(client, agent["prompt"], args.agent_max_tokens,
                          label=f"replay{i}")
        res["expected_code"] = agent["expected_code"]
        res["return_hit_pct"] = (
            round(100.0 * res["hits_delta"] / res["prompt_tokens"], 3)
            if res["prompt_tokens"] else None
        )
        replays.append(res)

    chaff_prompt_tokens = sum(c["prompt_tokens"] or 0 for c in chaff)
    return {
        "arm": arm,
        "chaff_no_store": no_store,
        "agents": [{k: v for k, v in a.items() if k != "prompt"} for a in agents],
        "agent_prompts": [a["prompt"] for a in agents],
        "chaff": chaff,
        "chaff_wall_s": round(chaff_wall, 3),
        "chaff_prompt_tokens": chaff_prompt_tokens,
        "chaff_prompt_tok_s": round(chaff_prompt_tokens / chaff_wall, 2) if chaff_wall else None,
        "replays": replays,
        "mean_return_hit_pct": round(
            statistics.fmean(r["return_hit_pct"] for r in replays
                             if r["return_hit_pct"] is not None), 3
        ) if any(r["return_hit_pct"] is not None for r in replays) else None,
    }


def case_retention(client: Client, args, rng: random.Random) -> dict:
    arms = []
    for no_store in args.arms:
        arms.append(retention_arm(client, args, rng, no_store))
    by_name = {a["arm"]: a for a in arms}
    return {
        "arms": arms,
        "verdict": verdict_retention(by_name.get("control"), by_name.get("treatment"),
                                    args),
    }


# ------------------------------------------------------------------ verdicts


def verdict_attribution(att: dict, args) -> dict:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    cases = att["cases"]
    cold = cases["cold"]
    add("cold_field_present", cold["details_present"],
        f"prompt_tokens_details present={cold['details_present']} usage={cold['usage_present']}")
    add("cold_cached_zero", cold["cached_tokens"] == 0,
        f"cached_tokens={cold['cached_tokens']} prompt_tokens={cold['prompt_tokens']}")
    add("cold_no_error", cold["error"] is None, str(cold["error"]))

    replay = cases["replay"]
    add("replay_hit_positive",
        isinstance(replay["cached_tokens"], int) and replay["cached_tokens"] > 0,
        f"cached_tokens={replay['cached_tokens']} prompt_tokens={replay['prompt_tokens']}")
    add("replay_counter_agrees", replay["cached_tokens"] == replay["hits_delta"],
        f"cached_tokens={replay['cached_tokens']} hits_delta={replay['hits_delta']}")

    negative = cases["negative"]
    add("negative_zero", negative["cached_tokens"] == 0,
        f"cached_tokens={negative['cached_tokens']} prompt_tokens={negative['prompt_tokens']}")
    add("negative_counter_agrees", negative["hits_delta"] == 0,
        f"hits_delta={negative['hits_delta']}")

    sse = cases["replay_sse"]
    add("sse_matches_http", sse["cached_tokens"] == replay["cached_tokens"],
        f"sse={sse['cached_tokens']} http={replay['cached_tokens']}")
    add("sse_counter_agrees", sse["cached_tokens"] == sse["hits_delta"],
        f"cached_tokens={sse['cached_tokens']} hits_delta={sse['hits_delta']}")
    add("sse_usage_present", sse["usage_present"],
        f"final usage chunk present={sse['usage_present']}")

    ov = att["overload"]
    add("overload_completed", ov["completed"] == ov["requests"],
        f"{ov['completed']}/{ov['requests']} completed")
    add("overload_no_errors", not ov["errors"], "; ".join(ov["errors"]) or "none")
    add("overload_observed_waiting", ov["max_waiting"] > 0 or ov["max_running"] <= 4,
        f"max_waiting={ov['max_waiting']} max_running={ov['max_running']} "
        f"max_waiting_capacity={ov['max_waiting_capacity']}")
    add("no_nan", not any(c["nan"] for c in cases.values()),
        "no NaN/locklock marker in the isolated cases")

    return {
        "gate": "attribution",
        "checks": checks,
        "passed": all(c["ok"] for c in checks),
    }


def verdict_retention(control: dict | None, treatment: dict | None, args) -> dict:
    if not control or not treatment:
        return {"gate": "retention", "passed": False,
                "checks": [{"check": "arms_present", "ok": False,
                            "detail": "control and treatment arms are both required"}]}
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str, skipped: bool = False) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail,
                       **({"skipped": True} if skipped else {})})

    c_pct, t_pct = control["mean_return_hit_pct"], treatment["mean_return_hit_pct"]
    gain = None if c_pct is None or t_pct is None else round(t_pct - c_pct, 3)
    add("return_hit_gain_pp",
        gain is not None and gain >= RETENTION_MIN_GAIN_PP,
        f"control={c_pct}% treatment={t_pct}% gain={gain}pp "
        f"threshold={RETENTION_MIN_GAIN_PP}pp")

    for arm in (control, treatment):
        errs = [r["error"] for r in arm["chaff"] + arm["replays"] if r["error"]]
        add(f"{arm['arm']}_no_request_errors", not errs, "; ".join(errs) or "none")

    for arm in (control, treatment):
        stale = [r["label"] for r in arm["replays"]
                 if r["answer_code"] != r["expected_code"]]
        add(f"{arm['arm']}_replays_return_own_code", not stale,
            f"mismatched replays: {stale or 'none'}")

    for arm in (control, treatment):
        wrong = [r["label"] for r in arm["chaff"]
                 if r["answer_code"] != r["expected_code"]]
        add(f"{arm['arm']}_chaff_answers_correct", not wrong,
            f"mismatched chaff: {wrong or 'none'}")

    for arm in (control, treatment):
        add(f"{arm['arm']}_no_nan",
            not any(r["nan"] for r in arm["chaff"] + arm["replays"]),
            "no NaN/locklock marker in chaff or replays")

    c_tp, t_tp = control["chaff_prompt_tok_s"], treatment["chaff_prompt_tok_s"]
    ratio = None if not c_tp or not t_tp else round(t_tp / c_tp, 4)
    add("chaff_throughput_floor",
        ratio is not None and ratio >= CHAFF_THROUGHPUT_FLOOR,
        f"control={c_tp} tok/s treatment={t_tp} tok/s ratio={ratio} "
        f"floor={CHAFF_THROUGHPUT_FLOOR}")

    for lane, floor in (args.floor or {}).items():
        val = (args.decode or {}).get(lane)
        if val is None:
            add(f"decode_floor_{lane}", True,
                f"skipped: no --decode-receipt for this lane (floor {floor} tok/s)",
                skipped=True)
            continue
        add(f"decode_floor_{lane}", val >= floor,
            f"median={val} tok/s floor={floor} tok/s")

    return {"gate": "retention", "checks": checks,
            "passed": all(c["ok"] for c in checks if not c.get("skipped")),
            "return_hit_gain_pp": gain}


def read_decode_receipts(paths: list[str]) -> dict[str, float]:
    """Lane -> median tok/s from `tests/bench_decode.py` receipts."""
    lanes: dict[str, float] = {}
    for p in paths or []:
        try:
            rec = json.loads(Path(p).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        phase = str(rec.get("phase") or Path(p).stem).lower()
        if "structured" in phase:
            lane = "structured"
        elif "essay" in phase:
            lane = "essay"
        else:
            lane = "prose"
        val = rec.get("tok_s_median")
        if isinstance(val, (int, float)):
            lanes[lane] = float(val)
    return lanes


def parse_floors(raw: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        lane, _, val = item.partition("=")
        try:
            out[lane.strip()] = float(val)
        except ValueError:
            raise SystemExit(f"--floor expects lane=tok_s pairs, got {item!r}")
    return out


def standing_controls(client: Client) -> dict:
    """Read-only snapshot of the controls this instrument must not move."""
    try:
        m = client.metrics()
    except OSError as exc:
        return {"error": str(exc)}
    return {
        "prefix_cache_hits_total": m.get("vllm:prefix_cache_hits_total"),
        "prefix_cache_queries_total": m.get("vllm:prefix_cache_queries_total"),
        "external_prefix_cache_queries_total": m.get("vllm:external_prefix_cache_queries_total"),
        "kv_cache_usage_perc": m.get("vllm:kv_cache_usage_perc"),
    }


# ---------------------------------------------------------------------- main


def run_boundary(args) -> dict:
    import subprocess
    script = Path(__file__).resolve().parent / "probe_apc_boundary_reachability.py"
    if not script.is_file():
        return {"gate": "boundary", "ran": False, "passed": False,
                "detail": f"{script} not found"}
    out = str(Path(args.out) / "apc-boundary.json")
    proc = subprocess.run([sys.executable, str(script), "--base", args.base,
                           "--out", out], capture_output=True, text=True)
    return {"gate": "boundary", "ran": True, "passed": proc.returncode == 0,
            "returncode": proc.returncode, "out": out,
            "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--out", required=True, help="receipt directory")
    ap.add_argument("--mode", default="all",
                    choices=["attribution", "retention", "boundary", "all"])
    ap.add_argument("--salt", default=None, help="unique run salt (default: random)")
    ap.add_argument("--seed", type=int, default=None, help="word-salad RNG seed")
    ap.add_argument("--attr-tokens", type=int, default=12000)
    ap.add_argument("--attr-max-tokens", type=int, default=24)
    ap.add_argument("--overload-requests", type=int, default=6,
                    help="concurrent requests; must exceed MAX_NUM_SEQS=4")
    ap.add_argument("--agent-count", type=int, default=2)
    ap.add_argument("--agent-tokens", type=int, default=46500)
    ap.add_argument("--agent-max-tokens", type=int, default=24)
    ap.add_argument("--chaff-count", type=int, default=18)
    ap.add_argument("--chaff-tokens", type=int, default=31000)
    ap.add_argument("--chaff-max-tokens", type=int, default=24)
    ap.add_argument("--needle-depth", type=int, default=512)
    ap.add_argument("--arms", default="0,1",
                    help="retention arms as chaff no-store bits, in run order")
    ap.add_argument("--decode-receipt", action="append", default=[],
                    help="tests/bench_decode.py receipt (repeatable)")
    ap.add_argument("--floor", default="structured=68.8,prose=30,essay=20")
    args = ap.parse_args(argv)

    args.salt = args.salt or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.arms = [a.strip() == "1" for a in args.arms.split(",") if a.strip()]
    args.floor = parse_floors(args.floor)
    args.decode = read_decode_receipts(args.decode_receipt)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed if args.seed is not None else int(time.time()))
    client = Client(args.base, args.api_key)

    receipt: dict = {
        "instrument": "probe_agentic_cache",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "salt": args.salt,
        "base": args.base,
        "arguments": {k: v for k, v in vars(args).items() if k not in {"api_key"}},
        "standing_controls": standing_controls(client),
        "gates": {},
    }

    if args.mode in ("attribution", "all"):
        att = case_attribution(client, args, rng)
        receipt["attribution"] = att
        receipt["gates"]["attribution"] = verdict_attribution(att, args)
        print(json.dumps(receipt["gates"]["attribution"], indent=2), flush=True)

    if args.mode in ("retention", "all"):
        ret = case_retention(client, args, rng)
        # The agent prompts are long; keep them in the receipt but not on stdout.
        receipt["retention"] = {
            k: v for k, v in ret.items() if k != "arms"
        }
        receipt["retention"]["arms"] = [
            {k: v for k, v in arm.items() if k != "agent_prompts"} for arm in ret["arms"]
        ]
        receipt["gates"]["retention"] = ret["verdict"]
        print(json.dumps(ret["verdict"], indent=2), flush=True)

    if args.mode in ("boundary", "all"):
        receipt["gates"]["boundary"] = run_boundary(args)

    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["passed"] = all(g.get("passed", False)
                            for g in receipt["gates"].values())
    path = out_dir / "agentic-cache.json"
    path.write_text(json.dumps(receipt, indent=2))
    print(f"\nreceipt: {path}")
    print(f"PASSED: {receipt['passed']}")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
