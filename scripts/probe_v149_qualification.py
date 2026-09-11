#!/usr/bin/env python3
"""One measurement block for the task 35 §6 qualification window.

Task 35 deployed the ExLlamaV3 v1.4.9 pin on the correctness gates, but the
end-to-end throughput comparison was never run to `docs/13` §6's observation
contract. This probe produces the observations that contract requires: fixed
saved payloads, identical sampling parameters, streaming, and per-run records
that keep the raw numbers so the auditor can re-judge them offline.

Kinds
-----
`structured` / `essay` / `hashmap`
    Streaming decode blocks (temp 0, top_p 1, thinking off, 200 output tokens).
    Decode tok/s = ``(completion_tokens - 1) / (t_last_token - t_first_token)``,
    the definition used by every prior receipt in this repository. The
    speculative-decoding counters are sampled around each run so the arm also
    carries an acceptance record (they are a *correctness* companion to the
    speed number, not a speed claim).

`prefill60k` / `prefill240k`
    Cold-prefill blocks. Every request carries a fresh salt, the filler count
    is calibrated through ``/tokenize`` so the server-reported ``prompt_tokens``
    lands on the target, and a run whose ``prompt_tokens_details.cached_tokens``
    is non-zero is rejected as warm. prefill tok/s = ``prompt_tokens / ttft``.

A run that cannot be measured is recorded and listed in ``invalid_runs``; it is
never silently dropped, because the auditor's sample-count gate reads that list.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE = os.environ.get("GLM53_BASE", "http://127.0.0.1:8000")
DEFAULT_MODEL = "GLM-5.3-Flash-EXL3"

STRUCTURED_PROMPT = (
    "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
)
ESSAY_PROMPT = (
    'Write a detailed technical essay titled "Speculative Decoding and the Hidden '
    'Cost of Failed Drafts: A Technical Analysis". Cover draft generation, '
    "verification cost, rejection sampling, and when longer drafts stop paying. "
    "Use numbered sections. Be thorough."
)
HASHMAP_PROMPT = (
    "Write a detailed step-by-step explanation of how a hash map works, "
    "including collision handling, resizing, and time complexity. Be thorough."
)
DECODE_KINDS = {
    "structured": STRUCTURED_PROMPT,
    "essay": ESSAY_PROMPT,
    "hashmap": HASHMAP_PROMPT,
}
PREFILL_KINDS = {"prefill60k": 60_000, "prefill240k": 240_000}
KINDS = tuple(DECODE_KINDS) + tuple(PREFILL_KINDS)
FILLER = "the "
TASK = "Reply with OK."
# Prompt-size acceptance band for a calibrated cold-prefill run.
PREFILL_TARGET_TOLERANCE = 0.05
NAN_RE = re.compile(r"\bnan\b|locklock", re.I)
SPEC_RE = re.compile(r"^(vllm:spec_decode_[a-zA-Z0-9_]+)\{([^}]*)\}\s+(\S+)$")


def contains_nan(text: str) -> bool:
    return bool(NAN_RE.search(text))


def _post(path: str, body: dict, timeout: float):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _get(path: str, timeout: float = 15.0) -> str:
    with urllib.request.urlopen(BASE + path, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def health() -> int:
    try:
        req = urllib.request.Request(BASE + "/health")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:  # noqa: BLE001  (an unreachable server is a health failure)
        return 0


def served_model() -> str:
    obj = json.loads(_get("/v1/models", timeout=30))
    return obj["data"][0]["id"]


def spec_snapshot() -> dict[str, float]:
    """Spec-decode counters; per-position keys are `pos:{n}`."""
    out: dict[str, float] = {}
    for line in _get("/metrics", timeout=15).splitlines():
        match = SPEC_RE.match(line)
        if not match:
            continue
        name, labels, value = match.group(1), match.group(2), float(match.group(3))
        if name.endswith("_created"):
            continue
        if "per_pos" in name:
            pos = re.search(r'position="(\d+)"', labels)
            if pos:
                out[f"pos:{pos.group(1)}"] = value
        else:
            out[name] = out.get(name, 0.0) + value
    return out


def spec_delta(before: dict[str, float], after: dict[str, float]) -> dict:
    drafts = after.get("vllm:spec_decode_num_drafts_total", 0.0) - before.get(
        "vllm:spec_decode_num_drafts_total", 0.0
    )
    draft_tokens = after.get("vllm:spec_decode_num_draft_tokens_total", 0.0) - before.get(
        "vllm:spec_decode_num_draft_tokens_total", 0.0
    )
    accepted = after.get("vllm:spec_decode_num_accepted_tokens_total", 0.0) - before.get(
        "vllm:spec_decode_num_accepted_tokens_total", 0.0
    )
    positions = []
    for index in range(7):
        key = f"pos:{index}"
        positions.append(
            round((after.get(key, 0.0) - before.get(key, 0.0)) / drafts, 4) if drafts else 0.0
        )
    return {
        "drafts": int(drafts),
        "draft_tokens": int(draft_tokens),
        "accepted": int(accepted),
        "accept_ratio": round(accepted / draft_tokens, 4) if draft_tokens else None,
        "accepted_per_step": round(accepted / drafts, 3) if drafts else None,
        "pos": positions,
    }


def _stream(body: dict, timeout: float) -> dict:
    """Drive one streaming completion and return raw timing + text.

    Reads the SSE stream **a line at a time**. A buffered `resp.read(4096)`
    blocks until 4096 bytes accumulate, which folds several token arrivals into
    one and destroys both the TTFT and the decode interval — a server flushing
    two tokens 250 ms apart was measured at ~6.2M tok/s because both landed in
    a single read and the interval collapsed to ~32 us. `readline()` returns
    each event as soon as the server flushes it.
    """
    started = time.perf_counter()
    first = None
    last = None
    chunks: list[str] = []
    usage = None
    finish = None
    http = None
    error = None
    try:
        with _post("/v1/chat/completions", body, timeout=timeout) as resp:
            http = resp.status
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                line = raw_line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = (
                    delta.get("content")
                    or delta.get("reasoning")
                    or delta.get("reasoning_content")
                    or ""
                )
                if content:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    chunks.append(content)
                if choices[0].get("finish_reason"):
                    finish = choices[0]["finish_reason"]
    except Exception as exc:  # noqa: BLE001  (a broken run is a recorded invalid run)
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    return {
        "http": http,
        "error": error,
        "first_s": first,
        "last_s": last,
        "ended_s": ended,
        "started_s": started,
        "text": "".join(chunks),
        "usage": usage or {},
        "finish_reason": finish,
    }


def decode_run(prompt: str, max_tokens: int, timeout: float) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    before = spec_snapshot()
    raw = _stream(body, timeout)
    after = spec_snapshot()
    usage = raw["usage"]
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    ttft = None if raw["first_s"] is None else raw["first_s"] - raw["started_s"]
    # First-to-last CONTENT token, not first-token-to-EOF. The usage chunk and
    # the connection close arrive after the final token; ending the interval at
    # EOF would charge that tail to decode.
    decode_s = (
        None
        if raw["first_s"] is None or raw["last_s"] is None
        else raw["last_s"] - raw["first_s"]
    )
    tok_s = None
    if decode_s and decode_s > 0 and completion_tokens > 1:
        tok_s = (completion_tokens - 1) / decode_s
    return {
        "http": raw["http"],
        "error": raw["error"],
        "ttft_s": ttft,
        "wall_s": raw["ended_s"] - raw["started_s"],
        "decode_s": decode_s,
        "tok_s": tok_s,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "finish_reason": raw["finish_reason"],
        "nan": contains_nan(raw["text"]),
        "text_head": raw["text"][:400],
        "spec": spec_delta(before, after),
    }


def decode_run_invalid(run: dict, max_tokens: int) -> str | None:
    if run["error"]:
        return run["error"]
    if run["http"] != 200:
        return f"http {run['http']}"
    if run["ttft_s"] is None:
        return "no first token"
    if run["tok_s"] is None:
        return f"completion_tokens={run['completion_tokens']} (need >1)"
    if run["nan"]:
        return "NaN/locklock marker in output"
    if run["completion_tokens"] < max_tokens:
        return f"short completion {run['completion_tokens']}/{max_tokens}"
    return None


def tokenize(messages: list[dict], timeout: float = 180.0) -> int:
    with _post("/tokenize", {"model": MODEL, "messages": messages}, timeout) as resp:
        return int(json.loads(resp.read().decode())["count"])


def build_user_text(n_filler: int, salt: str) -> str:
    return f"{salt}\n{FILLER * n_filler}\n{TASK}"


def calibrate(target: int, salt: str) -> tuple[int, int, str]:
    """Choose a filler count whose tokenize() lands within tolerance of target."""
    messages = lambda text: [{"role": "user", "content": text}]  # noqa: E731
    overhead = tokenize(messages(build_user_text(0, salt)))
    n = max(target - overhead, 1)
    got = tokenize(messages(build_user_text(n, salt)))
    if got > 0 and abs(got - target) / target > 0.005:
        per = (got - overhead) / n if n else 1.0
        n = max(int(round((target - overhead) / per)), 1) if per > 0 else n
        got = tokenize(messages(build_user_text(n, salt)))
    if abs(got - target) / target > 0.005:
        n = max(n + (target - got), 1)
        got = tokenize(messages(build_user_text(n, salt)))
    return n, got, build_user_text(n, salt)


def prefill_run(target: int, timeout: float) -> dict:
    salt = f"COLD-PREFILL salt={uuid.uuid4()} pad={secrets.token_hex(24)}"
    filler, estimate, text = calibrate(target, salt)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": 8,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    raw = _stream(body, timeout)
    usage = raw["usage"]
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    ttft = None if raw["first_s"] is None else raw["first_s"] - raw["started_s"]
    prefill_tok_s = prompt_tokens / ttft if ttft and ttft > 0 and prompt_tokens else None
    return {
        "http": raw["http"],
        "error": raw["error"],
        "target": target,
        "filler_count": filler,
        "tokenize_estimate": estimate,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "ttft_s": ttft,
        "wall_s": raw["ended_s"] - raw["started_s"],
        "prefill_tok_s": prefill_tok_s,
        "finish_reason": raw["finish_reason"],
        "nan": contains_nan(raw["text"]),
        "text_head": raw["text"][:120],
    }


def prefill_run_invalid(run: dict) -> str | None:
    if run["error"]:
        return run["error"]
    if run["http"] != 200:
        return f"http {run['http']}"
    if run["ttft_s"] is None:
        return "no first token"
    if not run["prompt_tokens"]:
        return "no prompt_tokens in usage"
    if run["prefill_tok_s"] is None:
        return "no prefill rate"
    target = run["target"]
    if abs(run["prompt_tokens"] - target) / target > PREFILL_TARGET_TOLERANCE:
        return f"prompt_tokens {run['prompt_tokens']} outside {target} +/- {PREFILL_TARGET_TOLERANCE:.0%}"
    if run["cached_tokens"] not in (0, None):
        return f"warm request: cached_tokens={run['cached_tokens']}"
    if run["nan"]:
        return "NaN/locklock marker in output"
    return None


def summarize(kind: str, runs: list[dict], invalid: list[dict]) -> dict:
    decode = kind in DECODE_KINDS
    key = "tok_s" if decode else "prefill_tok_s"
    values = [r[key] for r in runs if r.get(key) is not None and math.isfinite(r[key])]
    out: dict = {
        "schema": 1,
        "kind": kind,
        "metric": key,
        "base": BASE,
        "model": MODEL,
        "runs": runs,
        "invalid_runs": invalid,
        "valid_runs": len(values),
    }
    if values:
        out[f"{key}_median"] = statistics.median(values)
        out[f"{key}_min"] = min(values)
        out[f"{key}_max"] = max(values)
    if decode:
        ttfts = [r["ttft_s"] for r in runs if r.get("ttft_s") is not None]
        out["ttft_median_s"] = statistics.median(ttfts) if ttfts else None
        steps = [
            r["spec"]["accepted_per_step"]
            for r in runs
            if r.get("spec", {}).get("accepted_per_step") is not None
        ]
        ratios = [
            r["spec"]["accept_ratio"]
            for r in runs
            if r.get("spec", {}).get("accept_ratio") is not None
        ]
        out["accepted_per_step_median"] = statistics.median(steps) if steps else None
        out["accept_ratio_median"] = statistics.median(ratios) if ratios else None
        out["any_nan"] = any(r["nan"] for r in runs)
    else:
        out["any_cache_hit"] = any(r.get("cached_tokens") not in (0, None) for r in runs)
    return out


def write_receipt(path: Path, payload: dict) -> None:
    """Write the receipt atomically, so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    tmp.replace(path)


MODEL = DEFAULT_MODEL


def main(argv: list[str] | None = None) -> int:
    global MODEL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--runs", type=int, required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-tokens", type=int, default=200, help="decode block output tokens")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--base", default=None, help=f"API base (default {BASE})")
    ap.add_argument("--model", default=None, help="override the served model id")
    args = ap.parse_args(argv)
    if args.runs < 1:
        ap.error("--runs must be >= 1")
    if args.base:
        globals()["BASE"] = args.base

    code = health()
    if code != 200:
        print(f"[probe] /health -> {code}; refusing to measure", file=sys.stderr)
        return 2
    # The served id is authoritative; --model only overrides it when given.
    MODEL = args.model or served_model()

    runs: list[dict] = []
    invalid: list[dict] = []
    for index in range(args.runs):
        if args.kind in DECODE_KINDS:
            run = decode_run(DECODE_KINDS[args.kind], args.max_tokens, args.timeout)
            reason = decode_run_invalid(run, args.max_tokens)
        else:
            run = prefill_run(PREFILL_KINDS[args.kind], args.timeout)
            reason = prefill_run_invalid(run)
        run["i"] = index + 1
        if reason:
            run["invalid_reason"] = reason
            invalid.append(run)
        else:
            runs.append(run)
        shown = {
            k: run.get(k)
            for k in ("i", "tok_s", "prefill_tok_s", "ttft_s", "prompt_tokens", "cached_tokens")
        }
        print(f"[probe] {args.kind} run {index + 1}/{args.runs}: {json.dumps(shown)}"
              + (f" INVALID {reason}" if reason else ""), flush=True)
        # Persist after every observation: a crash in a later run must not
        # discard the observations already taken.
        write_receipt(args.out, summarize(args.kind, runs, invalid))

    summary = summarize(args.kind, runs, invalid)
    write_receipt(args.out, summary)
    headline = {k: v for k, v in summary.items() if k not in ("runs", "invalid_runs")}
    print(json.dumps(headline, indent=1, default=str), flush=True)
    print(f"[probe] wrote {args.out}", flush=True)
    return 0 if summary["valid_runs"] == args.runs else 1


if __name__ == "__main__":
    sys.exit(main())
