#!/usr/bin/env python3
"""Concurrency and long-context ladder for GLM-5.3-Flash EXL3.

Runs simultaneous streaming chat completions for each mode/context/concurrency
cell. Tokens come from the server usage block, not SSE chunk counts. Every cell
records aggregate and per-stream throughput, TTFT/ITL percentiles, prefix-cache
hit ratio, preemptions, request IDs, and enough counts for a server-log audit.

Canonical current-stack baseline, through the Mac tunnel:

  GLM53_BASE=http://127.0.0.1:8000 uv run python tests/bench_concurrency.py \
    --levels 1,2,3,4 --modes code,data,chat --ctx 0,60000 \
    --reps 3 --out local/concurrency-baseline-$(date +%F).json

The server must be idle unless ``--force`` is given. Long-context cells warm a
distinct system-message prefix per lane, then measure that cached prefix with a
different user suffix. Set ``VLLM_API_KEY`` for authenticated deployments.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

BASE = os.environ.get("GLM53_BASE", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("GLM53_MODEL", "GLM-5.3-Flash-EXL3")
API_KEY = os.environ.get("VLLM_API_KEY", "")

CHAT_TOPICS = [
    "why sourdough needs a long cold proof",
    "how tides work for a curious ten-year-old",
    "what makes a good cup of pour-over coffee",
    "how to plan a first vegetable garden in a small yard",
]
CODE_TASKS = [
    "a typed Python module implementing an LRU cache with tests",
    "a Go file implementing a thread-safe key-value store with TTL expiry",
    "a Rust tokenizer and recursive-descent parser for arithmetic expressions",
    "a bash log-rotation script with dry-run, verbose mode, and help text",
]
DATA_TASKS = [
    "a JSON array of 40 fictional customers with realistic fields",
    "a CSV with 60 fictional weather readings and a header",
    "an OpenAPI 3 YAML snippet for a four-endpoint notes API",
    "a markdown table comparing 45 fictional laptops",
]
FILLER = "Ledger row %d reconciled to the cent under audit rule seven. "
METRIC_NAMES = (
    "prefix_cache_queries_total",
    "prefix_cache_hits_total",
    "num_preemptions_total",
    "num_requests_running",
    "num_requests_waiting",
)


def _headers(*, json_content: bool = False, request_id: str = "") -> dict[str, str]:
    headers = {"Content-Type": "application/json"} if json_content else {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    if request_id:
        headers["X-Request-Id"] = request_id
    return headers


def _open(path: str, body: dict | None, timeout: float, request_id: str = ""):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        BASE + path,
        data=data,
        headers=_headers(json_content=body is not None, request_id=request_id),
        method="POST" if body is not None else "GET",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def parse_metrics(text: str) -> dict[str, float]:
    """Sum all label series for the counters and gauges used by the ladder."""
    out: dict[str, float] = {}
    for key in METRIC_NAMES:
        pattern = re.compile(rf"^vllm:{re.escape(key)}(?:\{{[^}}]*\}})?\s+(\S+)$", re.M)
        values = [float(match.group(1)) for match in pattern.finditer(text)]
        if values:
            out[key] = sum(values)
    return out


def metrics() -> dict[str, float]:
    with _open("/metrics", None, 20) as response:
        return parse_metrics(response.read().decode("utf-8", "replace"))


def server_is_busy(snapshot: dict[str, float]) -> bool:
    return (
        snapshot.get("num_requests_running", 0) > 0
        or snapshot.get("num_requests_waiting", 0) > 0
    )


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    ordered = sorted(values)

    def quantile(p: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
        return ordered[index]

    return {
        "p50": round(quantile(0.50), 3),
        "p95": round(quantile(0.95), 3),
        "p99": round(quantile(0.99), 3),
    }


def make_prefix(lane_tag: int, context_tokens: int) -> str:
    # Approximately fourteen tokens per sentence on this model's tokenizer.
    rows = max(0, int(context_tokens / 14))
    return "".join(FILLER % (lane_tag * 1_000_000 + row) for row in range(rows))


def task_text(mode: str, index: int) -> str:
    if mode == "chat":
        return (
            f"Chat with me about {CHAT_TOPICS[index % len(CHAT_TOPICS)]}. "
            "Be warm and conversational, about 250 words."
        )
    if mode == "code":
        return f"Write {CODE_TASKS[index % len(CODE_TASKS)]}. Output only code."
    return f"Write {DATA_TASKS[index % len(DATA_TASKS)]}. Output only the data."


def build_messages(
    mode: str,
    index: int,
    context_tokens: int,
    lane_tag: int,
    shared_prefix: bool,
    *,
    warmup: bool = False,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if context_tokens:
        prefix_lane = 0 if shared_prefix else lane_tag
        messages.append(
            {
                "role": "system",
                "content": (
                    make_prefix(prefix_lane, context_tokens)
                    + "\nThis ledger is immutable reference context."
                ),
            }
        )
    user = "Acknowledge the reference context with OK." if warmup else task_text(mode, index)
    messages.append({"role": "user", "content": user})
    return messages


def stream_one(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    thinking: bool,
    request_id: str,
    out: dict[str, Any],
) -> None:
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    started = time.perf_counter()
    first = None
    last = None
    itl: list[float] = []
    usage = None
    finish_reason = None
    http_status = None
    try:
        with _open(
            "/v1/chat/completions", body, timeout=3600, request_id=request_id
        ) as response:
            http_status = response.status
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
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
                    elif last is not None:
                        itl.append(now - last)
                    last = now
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    except urllib.error.HTTPError as exc:
        http_status = exc.code
        out["error"] = f"HTTP {exc.code}: {exc.read(200)!r}"
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"[:240]
    ended = time.perf_counter()
    out.update(
        {
            "request_id": request_id,
            "http": http_status,
            "start": started,
            "first": first,
            "end": ended,
            "ttft": (first - started) if first is not None else None,
            "itl": itl,
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "usage_seen": usage is not None,
            "finish_reason": finish_reason,
        }
    )


def _request_id(run_id: str, mode: str, ctx: int, level: int, rep: int, lane: int, kind: str) -> str:
    return f"glm53-ladder-{run_id}-{mode}-c{ctx}-n{level}-r{rep}-l{lane}-{kind}"


def run_cell(
    mode: str,
    ctx: int,
    level: int,
    rep: int,
    args: argparse.Namespace,
    run_id: str,
    status_cb,
) -> dict[str, Any]:
    temperature = 0.0 if mode in ("code", "data") or args.chat_temp0 else 0.7
    measured_messages = [
        build_messages(
            mode,
            rep * 31 + lane,
            ctx,
            lane + 1,
            args.shared_prefix,
        )
        for lane in range(level)
    ]
    warm_request_ids: list[str] = []
    if ctx:
        for lane in range(level):
            request_id = _request_id(run_id, mode, ctx, level, rep, lane, "warm")
            warm_request_ids.append(request_id)
            warm_out: dict[str, Any] = {}
            stream_one(
                build_messages(
                    mode,
                    rep * 31 + lane,
                    ctx,
                    lane + 1,
                    args.shared_prefix,
                    warmup=True,
                ),
                1,
                0.0,
                False,
                request_id,
                warm_out,
            )
            if warm_out.get("error") or warm_out.get("http") != 200:
                return {
                    "mode": mode,
                    "ctx": ctx,
                    "level": level,
                    "rep": rep,
                    "errors": 1,
                    "error_samples": [warm_out.get("error") or f"warmup HTTP {warm_out.get('http')}"],
                    "audit": {
                        "warm_request_ids": warm_request_ids,
                        "measured_request_ids": [],
                    },
                }
        time.sleep(1)

    before = metrics()
    cell_started = time.perf_counter()
    outputs: list[dict[str, Any]] = [dict() for _ in range(level)]
    request_ids = [
        _request_id(run_id, mode, ctx, level, rep, lane, "measure")
        for lane in range(level)
    ]
    threads = []
    for lane, messages in enumerate(measured_messages):
        thread = threading.Thread(
            target=stream_one,
            args=(
                messages,
                args.max_tokens,
                temperature,
                args.thinking,
                request_ids[lane],
                outputs[lane],
            ),
        )
        threads.append(thread)
        thread.start()
        if args.stagger and lane < level - 1:
            time.sleep(args.stagger)
    while any(thread.is_alive() for thread in threads):
        time.sleep(0.5)
        status_cb(mode, ctx, level, rep, outputs, cell_started)
    for thread in threads:
        thread.join()
    after = metrics()

    wall = max((item.get("end") or cell_started) for item in outputs) - cell_started
    tokens = [int(item.get("completion_tokens") or 0) for item in outputs]
    stream_rates = [
        (int(item["completion_tokens"]) - 1) / max(item["end"] - item["first"], 1e-3)
        for item in outputs
        if item.get("first") is not None and int(item.get("completion_tokens") or 0) > 1
    ]
    ttfts = [item["ttft"] for item in outputs if item.get("ttft") is not None]
    itls = [gap for item in outputs for gap in item.get("itl", [])]
    queries = after.get("prefix_cache_queries_total", 0) - before.get(
        "prefix_cache_queries_total", 0
    )
    hits = after.get("prefix_cache_hits_total", 0) - before.get(
        "prefix_cache_hits_total", 0
    )
    errors = [item.get("error") for item in outputs if item.get("error")]
    successful = sum(
        item.get("http") == 200 and item.get("usage_seen") for item in outputs
    )
    cell = {
        "mode": mode,
        "ctx": ctx,
        "level": level,
        "rep": rep,
        "temperature": temperature,
        "thinking": args.thinking,
        "agg_tps": round(sum(tokens) / max(wall, 1e-3), 1),
        "stream_tps_median": (
            round(statistics.median(stream_rates), 1) if stream_rates else None
        ),
        "ttft": percentiles(ttfts),
        "itl": percentiles(itls),
        "starvation_age_s": round(max(ttfts), 2) if ttfts else None,
        "tokens": sum(tokens),
        "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in outputs),
        "wall": round(wall, 1),
        "cache_hit_ratio": round(hits / queries, 3) if queries else None,
        "preemptions": int(
            after.get("num_preemptions_total", 0)
            - before.get("num_preemptions_total", 0)
        ),
        "errors": level - successful,
        "error_samples": errors[:2],
        "audit": {
            "warm_request_ids": warm_request_ids,
            "measured_request_ids": request_ids,
            "expected_measured_requests": level,
            "http_200_with_usage": successful,
            "metric_queries_delta": queries,
            "metric_hits_delta": hits,
        },
    }
    if ctx and cell["cache_hit_ratio"] is not None and cell["cache_hit_ratio"] < args.min_hit_ratio:
        cell["warning"] = (
            f"cache hit ratio {cell['cache_hit_ratio']} < {args.min_hit_ratio}"
        )
    return cell


def _positive_csv(raw: str, *, allow_zero: bool = False) -> list[int]:
    values = [int(part) for part in raw.split(",")]
    floor = 0 if allow_zero else 1
    if not values or any(value < floor for value in values):
        raise argparse.ArgumentTypeError(f"values must be >= {floor}: {raw}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--levels", default="1,2,3,4")
    parser.add_argument("--modes", default="code,data,chat")
    parser.add_argument("--ctx", default="0,60000", help="context tokens per lane")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--stagger", type=float, default=0.0)
    parser.add_argument("--shared-prefix", action="store_true")
    parser.add_argument("--chat-temp0", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--min-hit-ratio", type=float, default=0.8)
    parser.add_argument("--spread-tolerance", type=float, default=0.10)
    parser.add_argument("--status", default="status.json")
    parser.add_argument("--out", default="local/concurrency-ladder.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.reps < 1 or args.max_tokens < 1 or args.stagger < 0:
        parser.error("--reps/--max-tokens must be positive and --stagger non-negative")
    levels = _positive_csv(args.levels)
    contexts = _positive_csv(args.ctx, allow_zero=True)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if not modes or any(mode not in {"code", "data", "chat"} for mode in modes):
        parser.error("--modes must be a comma-separated subset of code,data,chat")

    initial = metrics()
    if not args.force and server_is_busy(initial):
        print(
            "server busy "
            f"(running={initial.get('num_requests_running', 0):g} "
            f"waiting={initial.get('num_requests_waiting', 0):g}); refusing. "
            "Use --force only when the overlap is intentional.",
            file=sys.stderr,
        )
        return 2

    run_id = uuid.uuid4().hex[:10]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.status)
    state: dict[str, Any] = {
        "schema_version": 1,
        "phase": "running",
        "run_id": run_id,
        "started": time.time(),
        "cells": [],
        "live": {},
    }

    def dump_status() -> None:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = status_path.with_name(status_path.name + ".tmp")
        temporary.write_text(json.dumps(state), encoding="utf-8")
        os.replace(temporary, status_path)

    def status_cb(mode, ctx, level, rep, outputs, cell_started) -> None:
        state["live"] = {
            "mode": mode,
            "ctx": ctx,
            "level": level,
            "rep": rep,
            "elapsed": round(time.perf_counter() - cell_started, 1),
            "streams": [
                {
                    "request_id": item.get("request_id"),
                    "ttft": item.get("ttft"),
                    "done": "end" in item,
                }
                for item in outputs
            ],
        }
        dump_status()

    warm: dict[str, Any] = {}
    stream_one(
        build_messages("code", 0, 0, 1, False),
        64,
        0.0,
        False,
        f"glm53-ladder-{run_id}-global-warm",
        warm,
    )
    if warm.get("error") or warm.get("http") != 200:
        print(f"global warmup failed: {warm}", file=sys.stderr)
        return 2

    for mode in modes:
        for ctx in contexts:
            for level in levels:
                measured: list[dict[str, Any]] = []
                for rep in range(args.reps):
                    cell = run_cell(
                        mode, ctx, level, rep, args, run_id, status_cb
                    )
                    measured.append(cell)
                    state["cells"].append(cell)
                    dump_status()
                    print(
                        f"{mode:5s} ctx={ctx:6d} x{level:2d} rep{rep}: "
                        f"agg {cell.get('agg_tps')} | "
                        f"stream {cell.get('stream_tps_median')} | "
                        f"TTFT p50/p95 {cell.get('ttft', {}).get('p50')}/"
                        f"{cell.get('ttft', {}).get('p95')} | "
                        f"hit {cell.get('cache_hit_ratio')} | "
                        f"pre {cell.get('preemptions')} | err {cell.get('errors')}",
                        flush=True,
                    )
                    time.sleep(2)
                aggregates = [
                    cell["agg_tps"] for cell in measured if cell.get("agg_tps")
                ]
                if (
                    len(aggregates) >= 2
                    and (max(aggregates) - min(aggregates))
                    / max(statistics.median(aggregates), 1e-3)
                    > args.spread_tolerance
                ):
                    cell = run_cell(
                        mode, ctx, level, args.reps, args, run_id, status_cb
                    )
                    cell["rerun"] = True
                    state["cells"].append(cell)
                    dump_status()
                    print(
                        f"  spread > {args.spread_tolerance:.0%}; "
                        f"re-ran once: agg {cell.get('agg_tps')}",
                        flush=True,
                    )

    state["phase"] = "done"
    state["finished"] = time.time()
    state["live"] = {}
    dump_status()
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "base": BASE,
        "model": MODEL,
        "args": vars(args),
        "initial_metrics": initial,
        "cells": state["cells"],
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 1 if any(cell.get("errors", 0) for cell in state["cells"]) else 0


if __name__ == "__main__":
    sys.exit(main())
