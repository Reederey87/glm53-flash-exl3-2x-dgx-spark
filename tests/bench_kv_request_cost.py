#!/usr/bin/env python3
"""Estimate fixed and marginal KV occupancy for one live request.

The gauge is sampled while one cold request prefills and decodes. The peak
``vllm:kv_cache_usage_perc`` value is converted with the boot-reported pool
token count, then a least-squares line estimates fixed request cost (intercept)
and marginal allocated pool tokens per prompt token (slope).

Canonical C2 probe:

  GLM53_BASE=http://127.0.0.1:8000 uv run python tests/bench_kv_request_cost.py \
    --contexts 2000,9000,30000,70000 --reps 3 --pool-tokens 1396551 \
    --out local/c2-fixed-kv-control-20260905.json
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
import urllib.request
import uuid
from pathlib import Path
from typing import Any

BASE = os.environ.get("GLM53_BASE", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("GLM53_MODEL", "GLM-5.3-Flash-EXL3")
API_KEY = os.environ.get("VLLM_API_KEY", "")
FILLER = "Ledger row %d reconciled to the cent under audit rule seven. "
METRICS = (
    "kv_cache_usage_perc",
    "num_preemptions_total",
    "num_requests_running",
    "num_requests_waiting",
)


def headers(request_id: str = "") -> dict[str, str]:
    result = {"Content-Type": "application/json"}
    if API_KEY:
        result["Authorization"] = f"Bearer {API_KEY}"
    if request_id:
        result["X-Request-Id"] = request_id
    return result


def open_url(path: str, body: dict | None, timeout: float, request_id: str = ""):
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers=headers(request_id),
        method="POST" if body is not None else "GET",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def parse_metrics(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in METRICS:
        values = re.findall(
            rf"^vllm:{re.escape(name)}(?:\{{[^}}]*\}})?\s+(\S+)$",
            text,
            re.MULTILINE,
        )
        if values:
            result[name] = sum(float(value) for value in values)
    return result


def metrics() -> dict[str, float]:
    with open_url("/metrics", None, 20) as response:
        return parse_metrics(response.read().decode("utf-8", "replace"))


def unique_text(approx_tokens: int, seed: int) -> str:
    # Calibrated on the production tokenizer: the unique row ids make this
    # approximately 19.5 tokens per sentence.
    rows = max(1, int(approx_tokens / 19.5))
    return f"[kv-salt {seed:x}] " + "".join(
        FILLER % (seed * 1_000_000 + row) for row in range(rows)
    )


def stream_request(
    context: int,
    seed: int,
    max_tokens: int,
    timeout: float,
    request_id: str,
    out: dict[str, Any],
) -> None:
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": unique_text(context, seed)
                + "\nWrite a detailed numbered analysis and continue until the limit.",
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "vllm_xargs": {"skip_writing_prefix_cache": 1},
    }
    started = time.perf_counter()
    first = None
    usage: dict[str, Any] = {}
    status = None
    done = False
    finish_reason = None
    try:
        with open_url(
            "/v1/chat/completions", body, timeout, request_id=request_id
        ) as response:
            status = response.status
            for raw in response:
                line = raw.decode("utf-8", "strict").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    continue
                event = json.loads(payload)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices:
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    finish_reason = choice.get("finish_reason") or finish_reason
                    if first is None and (
                        delta.get("content")
                        or delta.get("reasoning")
                        or delta.get("reasoning_content")
                    ):
                        first = time.perf_counter()
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    ended = time.perf_counter()
    out.update(
        {
            "request_id": request_id,
            "http": status,
            "ttft_s": first - started if first else None,
            "wall_s": ended - started,
            "usage": usage,
            "finish_reason": finish_reason,
            "done": done,
        }
    )


def request_failure(result: dict[str, Any]) -> str | None:
    if result.get("error"):
        return str(result["error"])
    if result.get("http") != 200:
        return f"HTTP {result.get('http')!r}"
    if not result.get("done"):
        return "missing SSE [DONE]"
    if result.get("finish_reason") not in {"stop", "length"}:
        return f"finish_reason={result.get('finish_reason')!r}"
    usage = result.get("usage") or {}
    if not usage.get("prompt_tokens") or not usage.get("completion_tokens"):
        return "missing final usage"
    return None


def wait_idle(timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = metrics()
        if (
            snapshot.get("num_requests_running", 0) == 0
            and snapshot.get("num_requests_waiting", 0) == 0
            and snapshot.get("kv_cache_usage_perc", 0) == 0
        ):
            return True
        time.sleep(0.5)
    return False


def fit_line(points: list[tuple[float, float]]) -> dict[str, float | int | None]:
    if len(points) < 2:
        return {"n": len(points), "intercept": None, "slope": None, "r2": None}
    mean_x = statistics.mean(point[0] for point in points)
    mean_y = statistics.mean(point[1] for point in points)
    ss_x = sum((x - mean_x) ** 2 for x, _ in points)
    if ss_x == 0:
        return {"n": len(points), "intercept": None, "slope": None, "r2": None}
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / ss_x
    intercept = mean_y - slope * mean_x
    predicted = [intercept + slope * x for x, _ in points]
    ss_res = sum((y - yhat) ** 2 for (_, y), yhat in zip(points, predicted))
    ss_tot = sum((y - mean_y) ** 2 for _, y in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return {
        "n": len(points),
        "intercept": round(intercept, 1),
        "slope": round(slope, 6),
        "r2": round(r2, 6),
    }


def run_probe(context: int, rep: int, args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    if not wait_idle(args.idle_timeout):
        return {"context": context, "rep": rep, "error": "engine did not drain"}
    before = metrics()
    request_id = f"glm53-c2-{run_id}-c{context}-r{rep}"
    request_out: dict[str, Any] = {}
    thread = threading.Thread(
        target=stream_request,
        args=(
            context,
            context * 1009 + rep,
            args.max_tokens,
            args.timeout,
            request_id,
            request_out,
        ),
    )
    samples: list[dict[str, float]] = []
    thread.start()
    while thread.is_alive():
        snapshot = metrics()
        samples.append(
            {
                "elapsed_s": time.perf_counter(),
                "kv_cache_usage_perc": snapshot.get("kv_cache_usage_perc", 0),
                "running": snapshot.get("num_requests_running", 0),
                "waiting": snapshot.get("num_requests_waiting", 0),
            }
        )
        time.sleep(args.poll_interval)
    thread.join()
    after = metrics()
    failure = request_failure(request_out)
    peak = max((row["kv_cache_usage_perc"] for row in samples), default=0)
    prompt_tokens = int((request_out.get("usage") or {}).get("prompt_tokens") or 0)
    completion_tokens = int((request_out.get("usage") or {}).get("completion_tokens") or 0)
    return {
        "context": context,
        "rep": rep,
        "error": failure,
        "request_id": request_id,
        "request": request_out,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "peak_kv_cache_usage_perc": peak,
        "estimated_pool_tokens": round(peak * args.pool_tokens, 1),
        "estimated_excess_over_prompt": round(
            peak * args.pool_tokens - prompt_tokens, 1
        ),
        "queue_peak_waiting": max((row["waiting"] for row in samples), default=0),
        "preemptions": int(
            after.get("num_preemptions_total", 0)
            - before.get("num_preemptions_total", 0)
        ),
        "samples": samples,
    }


def positive_csv(raw: str) -> list[int]:
    values = [int(part) for part in raw.split(",")]
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("contexts must be positive integers")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", default="2000,9000,30000,70000")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--pool-tokens", type=int, default=1_396_551)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--idle-timeout", type=float, default=120)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    contexts = positive_csv(args.contexts)
    if args.reps < 1 or args.max_tokens < 1 or args.pool_tokens < 1:
        parser.error("reps, max-tokens, and pool-tokens must be positive")
    initial = metrics()
    if not args.force and (
        initial.get("num_requests_running", 0) > 0
        or initial.get("num_requests_waiting", 0) > 0
    ):
        print("server busy; refusing (use --force only for intentional overlap)", file=sys.stderr)
        return 2

    run_id = uuid.uuid4().hex[:10]
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "started": time.time(),
        "base": BASE,
        "model": MODEL,
        "args": vars(args),
        "initial_metrics": initial,
        "probes": [],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for context in contexts:
        for rep in range(args.reps):
            row = run_probe(context, rep, args, run_id)
            result["probes"].append(row)
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(
                f"ctx={context} rep={rep + 1}/{args.reps} "
                f"prompt={row.get('prompt_tokens')} peak={row.get('peak_kv_cache_usage_perc')} "
                f"pool_tokens={row.get('estimated_pool_tokens')} "
                f"preemptions={row.get('preemptions', 0)} error={row.get('error')!r}",
                flush=True,
            )
            time.sleep(1)
    valid = [
        row
        for row in result["probes"]
        if not row.get("error") and row.get("prompt_tokens", 0) > 0
    ]
    points = [
        (float(row["prompt_tokens"]), float(row["estimated_pool_tokens"]))
        for row in valid
    ]
    result["fit"] = fit_line(points)
    result["context_medians"] = {
        str(context): {
            "prompt_tokens": round(
                statistics.median(
                    row["prompt_tokens"]
                    for row in valid
                    if row["context"] == context
                ),
                1,
            ),
            "estimated_pool_tokens": round(
                statistics.median(
                    row["estimated_pool_tokens"]
                    for row in valid
                    if row["context"] == context
                ),
                1,
            ),
            "estimated_excess_over_prompt": round(
                statistics.median(
                    row["estimated_excess_over_prompt"]
                    for row in valid
                    if row["context"] == context
                ),
                1,
            ),
        }
        for context in contexts
        if any(row["context"] == context for row in valid)
    }
    result["finished"] = time.time()
    result["final_metrics"] = metrics()
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"fit": result["fit"], "contexts": result["context_medians"]}, indent=2))
    return (
        1
        if any(
            row.get("error") or row.get("preemptions")
            for row in result["probes"]
        )
        else 0
    )


if __name__ == "__main__":
    sys.exit(main())
