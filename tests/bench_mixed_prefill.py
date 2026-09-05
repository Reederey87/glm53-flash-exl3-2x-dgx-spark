#!/usr/bin/env python3
"""Measure mixed-prefill newcomer latency against a running decode.

Each sample starts one incumbent generation, waits until it has emitted tokens,
then launches one cold newcomer after ``--arrival-delay`` seconds. The report
includes newcomer TTFT, incumbent decode rate and ITL before/during the overlap,
aggregate throughput, preemptions, queue peaks, and request IDs for journal
audit.

Canonical C1 arm:

  GLM53_BASE=http://127.0.0.1:8000 uv run python tests/bench_mixed_prefill.py \
    --contexts 9500,38000 --samples 100 \
    --out local/c1-ARM-newcomer-20260905.json
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
FILLER = "Ledger row %d reconciled to the cent under audit rule seven. "
METRICS = (
    "generation_tokens_total",
    "prompt_tokens_total",
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


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(quantile * (len(ordered) - 1)))))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "median": round(statistics.median(values), 3) if values else None,
        "p95": round(percentile(values, 0.95), 3) if values else None,
        "p99": round(percentile(values, 0.99), 3) if values else None,
        "min": round(min(values), 3) if values else None,
        "max": round(max(values), 3) if values else None,
    }


def unique_text(approx_tokens: int, seed: int) -> str:
    # The large unique row ids make this about 19.5 tokens per sentence on the
    # production tokenizer (calibrated by the cluster smoke before the C1 run).
    rows = max(1, int(approx_tokens / 19.5))
    return f"[cold-salt {seed:x}] " + "".join(
        FILLER % (seed * 1_000_000 + row) for row in range(rows)
    )


def stream_chat(
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    timeout: float,
    request_id: str,
    out: dict[str, Any],
    first_token_event: threading.Event | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "vllm_xargs": {"skip_writing_prefix_cache": 1},
    }
    started = time.perf_counter()
    token_events: list[dict[str, float | int]] = []
    usage: dict[str, Any] = {}
    finish_reason = None
    status = None
    done = False
    intentional_stop = False
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
                if not choices:
                    continue
                choice = choices[0]
                token_ids = choice.get("token_ids") or []
                if token_ids:
                    token_events.append(
                        {"time": time.perf_counter(), "count": len(token_ids)}
                    )
                    if first_token_event is not None:
                        first_token_event.set()
                    if stop_event is not None and stop_event.is_set():
                        intentional_stop = True
                        break
                finish_reason = choice.get("finish_reason") or finish_reason
    except Exception as exc:  # report the sample instead of losing the arm
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    ended = time.perf_counter()
    out.update(
        {
            "request_id": request_id,
            "http": status,
            "started": started,
            "ended": ended,
            "token_events": token_events,
            "ttft_s": (
                float(token_events[0]["time"]) - started if token_events else None
            ),
            "usage": usage,
            "finish_reason": finish_reason,
            "done": done,
            "intentional_stop": intentional_stop,
        }
    )


def token_rate(
    token_events: list[dict[str, float | int]], start: float, end: float
) -> float | None:
    if end <= start:
        return None
    count = sum(
        int(event["count"])
        for event in token_events
        if start <= float(event["time"]) <= end
    )
    return count / (end - start)


def token_arrival_gaps(
    token_events: list[dict[str, float | int]], start: float, end: float
) -> list[float]:
    """Return token-arrival gaps clipped to the observation window.

    Tokens delivered in one speculative batch have zero client-arrival gap.
    Leading/trailing silence is retained, so a stall crossing either window
    boundary cannot disappear from the reported distribution.
    """
    if end <= start:
        return []
    selected = [
        event
        for event in token_events
        if start <= float(event["time"]) <= end
    ]
    if not selected:
        return [end - start]
    gaps = [float(selected[0]["time"]) - start]
    for event in selected:
        gaps.extend([0.0] * max(0, int(event["count"]) - 1))
    gaps.extend(
        float(right["time"]) - float(left["time"])
        for left, right in zip(selected, selected[1:])
    )
    gaps.append(end - float(selected[-1]["time"]))
    return gaps


def request_failure(
    result: dict[str, Any], *, allow_intentional_stop: bool = False
) -> str | None:
    if result.get("error"):
        return str(result["error"])
    if result.get("http") != 200:
        return f"HTTP {result.get('http')!r}"
    if not result.get("token_events"):
        return "no returned token IDs"
    if allow_intentional_stop and result.get("intentional_stop"):
        return None
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
        ):
            return True
        time.sleep(1)
    return False


def run_sample(context: int, sample: int, args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    if not wait_idle(args.idle_timeout):
        return {"context": context, "sample": sample, "error": "engine did not drain"}
    before = metrics()
    first_token = threading.Event()
    stop_incumbent = threading.Event()
    incumbent: dict[str, Any] = {}
    newcomer: dict[str, Any] = {}
    incumbent_id = f"glm53-c1-{run_id}-c{context}-r{sample}-incumbent"
    newcomer_id = f"glm53-c1-{run_id}-c{context}-r{sample}-newcomer"
    incumbent_prompt = (
        "Write an extremely long, detailed maritime operations handbook. "
        "Continue with numbered sections and do not conclude early."
    )
    incumbent_thread = threading.Thread(
        target=stream_chat,
        kwargs={
            "prompt": incumbent_prompt,
            "max_tokens": args.incumbent_tokens,
            "temperature": 0.7,
            "timeout": args.timeout,
            "request_id": incumbent_id,
            "out": incumbent,
            "first_token_event": first_token,
            "stop_event": stop_incumbent,
        },
        daemon=True,
    )
    incumbent_thread.start()
    if not first_token.wait(args.incumbent_start_timeout):
        stop_incumbent.set()
        incumbent_thread.join(timeout=5)
        return {
            "context": context,
            "sample": sample,
            "error": "incumbent did not enter decode",
            "incumbent": incumbent,
        }
    decode_started = time.perf_counter()
    time.sleep(args.arrival_delay)
    newcomer_prompt = (
        unique_text(context, (sample + 1) * 1_000_003 + context)
        + "\nIgnore the reference and answer with exactly: NEWCOMER_OK"
    )
    queue_samples: list[dict[str, float]] = []
    newcomer_thread = threading.Thread(
        target=stream_chat,
        kwargs={
            "prompt": newcomer_prompt,
            "max_tokens": args.newcomer_tokens,
            "temperature": 0.0,
            "timeout": args.timeout,
            "request_id": newcomer_id,
            "out": newcomer,
        },
        daemon=True,
    )
    newcomer_thread.start()
    while newcomer_thread.is_alive():
        snapshot = metrics()
        queue_samples.append(
            {
                "t": time.perf_counter(),
                "running": snapshot.get("num_requests_running", 0),
                "waiting": snapshot.get("num_requests_waiting", 0),
            }
        )
        time.sleep(args.poll_interval)
    newcomer_thread.join()
    overlap_end = time.perf_counter()
    stop_incumbent.set()
    incumbent_thread.join()
    after = metrics()
    incumbent_failure = request_failure(
        incumbent, allow_intentional_stop=True
    )
    newcomer_failure = request_failure(newcomer)
    failure = None
    if incumbent_failure or newcomer_failure:
        failure = (
            f"incumbent: {incumbent_failure}" if incumbent_failure else ""
        )
        if newcomer_failure:
            failure = (
                f"{failure}; " if failure else ""
            ) + f"newcomer: {newcomer_failure}"
    token_events = incumbent.get("token_events") or []
    overlap_start = newcomer.get("started", overlap_end)
    before_start = max(decode_started, overlap_start - args.before_window)
    before_rate = token_rate(token_events, before_start, overlap_start)
    during_rate = token_rate(token_events, overlap_start, overlap_end)
    during_gaps = token_arrival_gaps(
        token_events, overlap_start, overlap_end
    )
    sample_wall = max(
        incumbent.get("ended", overlap_end), newcomer.get("ended", overlap_end)
    ) - min(incumbent.get("started", overlap_start), newcomer.get("started", overlap_start))
    generation_tokens_delta = (
        after.get("generation_tokens_total", 0)
        - before.get("generation_tokens_total", 0)
    )
    return {
        "context": context,
        "sample": sample,
        "error": failure,
        "incumbent_request_id": incumbent_id,
        "newcomer_request_id": newcomer_id,
        "incumbent": {
            key: value
            for key, value in incumbent.items()
            if key not in {"token_events", "started", "ended"}
        },
        "newcomer": {
            key: value
            for key, value in newcomer.items()
            if key not in {"token_events", "started", "ended"}
        },
        "incumbent_rate_before": round(before_rate, 3) if before_rate is not None else None,
        "incumbent_rate_during": round(during_rate, 3) if during_rate is not None else None,
        "incumbent_token_events": len(token_events),
        "incumbent_tokens_observed": sum(
            int(event["count"]) for event in token_events
        ),
        "incumbent_max_tokens_per_event": max(
            (int(event["count"]) for event in token_events), default=0
        ),
        "incumbent_rate_ratio": (
            round(during_rate / before_rate, 4)
            if before_rate and during_rate is not None
            else None
        ),
        "incumbent_token_arrival_gap_during": summarize(during_gaps),
        # The incumbent stream is deliberately cancelled after the newcomer, so
        # its final usage block is unavailable. The server counter is the only
        # complete output-token ledger for the isolated sample.
        "aggregate_tps": round(generation_tokens_delta / sample_wall, 3),
        "queue_peak_running": max((row["running"] for row in queue_samples), default=0),
        "queue_peak_waiting": max((row["waiting"] for row in queue_samples), default=0),
        "preemptions": int(
            after.get("num_preemptions_total", 0)
            - before.get("num_preemptions_total", 0)
        ),
        "metric_delta": {
            key: after.get(key, 0) - before.get(key, 0) for key in METRICS
        },
    }


def summarize_context(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in samples if not row.get("error")]
    newcomer_ttft = [
        float(row["newcomer"]["ttft_s"])
        for row in valid
        if row.get("newcomer", {}).get("ttft_s") is not None
    ]
    rate_ratio = [
        float(row["incumbent_rate_ratio"])
        for row in valid
        if row.get("incumbent_rate_ratio") is not None
    ]
    itl_p99 = [
        float(row["incumbent_token_arrival_gap_during"]["p99"])
        for row in valid
        if row.get("incumbent_token_arrival_gap_during", {}).get("p99")
        is not None
    ]
    aggregate = [
        float(row["aggregate_tps"])
        for row in valid
        if row.get("aggregate_tps") is not None
    ]
    return {
        "samples": len(samples),
        "valid": len(valid),
        "errors": len(samples) - len(valid),
        "newcomer_ttft_s": summarize(newcomer_ttft),
        "incumbent_rate_ratio": summarize(rate_ratio),
        "incumbent_token_arrival_gap_p99_s": summarize(itl_p99),
        "aggregate_tps": summarize(aggregate),
        "newcomer_prompt_tokens": summarize(
            [
                float(row["newcomer"]["usage"]["prompt_tokens"])
                for row in valid
                if row.get("newcomer", {}).get("usage", {}).get("prompt_tokens")
            ]
        ),
        "preemptions": sum(int(row.get("preemptions", 0)) for row in valid),
        "queue_peak_waiting": max(
            (float(row.get("queue_peak_waiting", 0)) for row in valid), default=0
        ),
    }


def positive_csv(raw: str) -> list[int]:
    values = [int(part) for part in raw.split(",")]
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("contexts must be positive integers")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", default="9500,38000")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--incumbent-tokens", type=int, default=2400)
    parser.add_argument("--newcomer-tokens", type=int, default=8)
    parser.add_argument("--arrival-delay", type=float, default=2.0)
    parser.add_argument("--before-window", type=float, default=2.0)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--idle-timeout", type=float, default=120)
    parser.add_argument("--incumbent-start-timeout", type=float, default=300)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    contexts = positive_csv(args.contexts)
    if args.samples < 1 or args.incumbent_tokens < 2 or args.newcomer_tokens < 1:
        parser.error("sample and token counts must be positive")
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
        "samples": [],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for context in contexts:
        for sample in range(args.samples):
            row = run_sample(context, sample, args, run_id)
            result["samples"].append(row)
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            ttft = row.get("newcomer", {}).get("ttft_s")
            print(
                f"ctx={context} sample={sample + 1}/{args.samples} "
                f"newcomer_ttft={ttft!r} rate_ratio={row.get('incumbent_rate_ratio')!r} "
                "gap_p99="
                f"{row.get('incumbent_token_arrival_gap_during', {}).get('p99')!r} "
                f"preemptions={row.get('preemptions', 0)} error={row.get('error')!r}",
                flush=True,
            )
            time.sleep(1)
    result["finished"] = time.time()
    result["summary"] = {
        str(context): summarize_context(
            [row for row in result["samples"] if row.get("context") == context]
        )
        for context in contexts
    }
    result["final_metrics"] = metrics()
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 1 if any(row.get("error") or row.get("preemptions") for row in result["samples"]) else 0


if __name__ == "__main__":
    sys.exit(main())
