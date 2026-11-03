#!/usr/bin/env python3
"""Supplementary client-observed prefill/decode overlap probe; no adoption gate.

Keep the mandatory full speed matrix unchanged. This probe reports gaps between
cumulative-usage updates, not individual token times or GPU co-execution.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request

HELPER = Path(__file__).resolve().parents[2] / "scripts/speed_matrix.py"
spec = importlib.util.spec_from_file_location("mixed_speed_helpers", HELPER)
speed = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = speed
spec.loader.exec_module(speed)
SUCCESS = "vllm:request_success_total"
OCCUPANCY = ("vllm:num_requests_running", "vllm:num_requests_waiting")
METRICS = (SUCCESS, speed.CACHE_HITS, speed.ACCEPTED, speed.DRAFTED, speed.DRAFTS, *OCCUPANCY)


@dataclass
class Trace:
    role: str
    prompt_sha256: str
    prompt_tokens: int
    budget: int
    started_s: float | None = None
    ended_s: float | None = None
    arrivals: list = field(default_factory=list)
    final_usage: dict | None = None
    finish_reason: str | None = None
    done: bool = False
    errors: list[str] = field(default_factory=list)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def stream(base, model, prompt, trace, origin, timeout, api_key="", notify=lambda: None):
    """Retain incomplete evidence and notify the coordinator on real token progress."""
    payload = {"model": model, "prompt": prompt, "max_tokens": trace.budget,
               "temperature": 0, "seed": 0, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(base.rstrip("/") + "/v1/completions",
                                     data=json.dumps(payload).encode(), headers=headers)
    trace.started_s = time.perf_counter() - origin
    count = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                if time.perf_counter() - origin - trace.started_s > timeout:
                    raise TimeoutError("request exceeded total time budget")
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    trace.done = True
                    break
                event = json.loads(body)
                if not isinstance(event, dict) or event.get("error"):
                    raise ValueError("SSE error or non-object event")
                choices = event.get("choices", [])
                usage = event.get("usage")
                if not isinstance(choices, list):
                    raise ValueError("invalid choices")
                if not choices:
                    if isinstance(usage, dict):
                        trace.final_usage = usage
                    continue
                if len(choices) != 1 or not isinstance(choices[0], dict):
                    raise ValueError("requires one choice")
                current = usage.get("completion_tokens") if isinstance(usage, dict) else None
                if (type(current) is not int or not count <= current <= trace.budget
                        or type(usage.get("prompt_tokens")) is not int
                        or usage["prompt_tokens"] != len(prompt)):
                    raise ValueError("invalid continuous token usage")
                if current > count:
                    if trace.finish_reason is not None:
                        raise ValueError("progress after finish")
                    trace.arrivals.append((current, time.perf_counter() - origin))
                    notify()
                count = current
                reason = choices[0].get("finish_reason")
                if reason is not None:
                    trace.finish_reason = reason
    except (ValueError, OSError, UnicodeError) as exc:
        trace.errors.append(str(exc))
    finally:
        trace.ended_s = time.perf_counter() - origin
        usage = trace.final_usage or {}
        if (not trace.done or trace.finish_reason != "length" or count != trace.budget
                or not trace.arrivals or type(usage.get("completion_tokens")) is not int
                or usage.get("completion_tokens") != trace.budget
                or type(usage.get("prompt_tokens")) is not int
                or usage.get("prompt_tokens") != len(prompt)):
            trace.errors.append("incomplete fixed-budget stream")
        notify()


def analyze(short: list[Trace], long: Trace) -> dict:
    errors = [f"{trace.role}: {error}" for trace in [*short, long] for error in trace.errors]
    rows = []
    for trace in [*short, long]:
        usage = trace.final_usage or {}
        previous_count, previous_time = 0, trace.started_s
        valid = (trace.done and trace.finish_reason == "length" and bool(trace.arrivals)
                 and type(trace.started_s) in (int, float) and math.isfinite(trace.started_s)
                 and type(trace.ended_s) in (int, float) and math.isfinite(trace.ended_s)
                 and type(usage.get("prompt_tokens")) is int
                 and type(usage.get("completion_tokens")) is int
                 and usage.get("prompt_tokens") == trace.prompt_tokens
                 and usage.get("completion_tokens") == trace.budget)
        if valid:
            for count, timestamp in trace.arrivals:
                valid &= (type(count) is int and previous_count < count <= trace.budget
                          and type(timestamp) in (int, float) and math.isfinite(timestamp)
                          and previous_time <= timestamp <= trace.ended_s)
                if not valid:
                    break
                previous_count, previous_time = count, timestamp
            valid &= previous_count == trace.budget
        if not valid:
            errors.append(f"{trace.role}: invalid or incomplete trace")
    if errors:
        return {"ok": False, "errors": errors, "short": rows}
    if long.started_s is None or not long.arrivals:
        return {"ok": False, "errors": [*errors, "long prefill never produced a token"], "short": rows}
    start, stop = long.started_s, long.arrivals[0][1]
    if not math.isfinite(stop - start) or stop <= start:
        return {"ok": False, "errors": [*errors, "invalid prefill interval"], "short": rows}
    for trace in short:
        arrivals = trace.arrivals
        before = [(n, t) for n, t in arrivals if t <= start]
        inside = [(n, t) for n, t in arrivals if start < t < stop]
        after = [(n, t) for n, t in arrivals if t > stop]
        # Two progress updates inside, plus progress on both sides, exclude
        # a queued-only or already-finished decode from this overlap probe.
        overlap = bool(before and len(inside) >= 2 and after)
        if not overlap:
            errors.append(f"{trace.role}: insufficient observed decode overlap")
        gaps = [b[1] - a[1] for a, b in zip(arrivals, arrivals[1:])
                if a[1] < stop and b[1] > start]
        delta = inside[-1][0] - before[-1][0] if before and inside else None
        rows.append({"role": trace.role, "observed_overlap": overlap,
                     "usage_updates_inside_prefill": len(inside),
                     "observed_tokens_inside_prefill": delta,
                     "observed_tokens_per_prefill_second": delta / (stop - start) if delta is not None else None,
                     "overlapping_update_gap_p95_s": speed.percentile(gaps, 0.95),
                     "overlapping_update_gap_max_s": max(gaps) if gaps else None})
    return {"ok": not errors, "errors": errors, "short": rows,
            "long_ttft_s": stop - start,
            "long_prefill_tokens_per_s": long.prompt_tokens / (stop - start)}


def counter_delta(before, after, name):
    values = [sample.get(name) for sample in (before, after)]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError(f"missing/invalid counter {name}")
    if values[1] < values[0]:
        raise ValueError(f"counter reset: {name}")
    return values[1] - values[0]


def metric_snapshot(sample):
    # Keep only relevant metrics; preserve nonfinite diagnostics as valid JSON.
    return {name: value if math.isfinite(value) else repr(value)
            for name in METRICS if (value := sample.get(name)) is not None}


def prompts_for(args, repetition, decoders, length):
    short = []
    for index in range(decoders):
        identity = [args.seed, "mixed-prefill-v1", repetition, decoders, length, index]
        payload = {"model": args.model, "add_generation_prompt": True,
                   "chat_template_kwargs": {"reasoning_effort": "max"},
                   "messages": [{"role": "user", "content":
                       f"Trial {fingerprint(identity)}. {speed.DECODE_TASKS['code']}"}]}
        _, response = speed._request_json(args.base.rstrip("/") + "/tokenize", payload, args.api_key, args.timeout)
        tokens = response.get("tokens") if isinstance(response, dict) else None
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("invalid tokenized maximum-reasoning prompt")
        short.append(tokens)
    long = speed.make_prompt(length, speed.prompt_rng(args.seed, "mixed-prefill-v1", repetition,
                                                     decoders, length, decoders))
    return short, long


def measure_cell(args, repetition, decoders, length, budget, cell):
    short_prompts, long_prompt = prompts_for(args, repetition, decoders, length)
    short = [Trace(f"decode-{i}", fingerprint(p), len(p), budget) for i, p in enumerate(short_prompts)]
    long = Trace("prefill", fingerprint(long_prompt), len(long_prompt), 1)
    cell.update(repetition=repetition, decoders=decoders, length=length, budget=budget,
                requests=[*short, long], ok=False, complete=False)
    before = speed.scrape_metrics(args.base, args.api_key, args.timeout)
    cell["metrics_before"] = metric_snapshot(before)
    if any(before.get(name) != 0 for name in OCCUPANCY):
        raise ValueError("server is not idle or occupancy metrics are missing before the cell")
    origin = time.perf_counter()
    cell["started_utc"] = datetime.now(timezone.utc).isoformat()
    cell["time_origin"] = "all request and update times are seconds from this cell's monotonic origin"
    condition = threading.Condition()

    def notify():
        with condition:
            condition.notify_all()

    def ready():
        return all(t.arrivals and args.ready_tokens <= t.arrivals[-1][0] < t.budget for t in short)

    with ThreadPoolExecutor(max_workers=decoders) as pool:
        futures = [pool.submit(stream, args.base, args.model, prompt, trace, origin,
                               args.timeout, args.api_key, notify)
                   for prompt, trace in zip(short_prompts, short)]
        with condition:
            signaled = condition.wait_for(lambda: ready() or any(t.ended_s is not None for t in short),
                                          timeout=args.ready_timeout)
            launch = signaled and ready() and all(t.ended_s is None for t in short)
        cell["ready_barrier_passed"] = launch
        if launch:
            stream(args.base, args.model, long_prompt, long, origin, args.timeout, args.api_key)
        else:
            long.errors.append("decoders did not reach readiness while still active")
        for future in futures:
            future.result()
    cell["wall_s"] = time.perf_counter() - origin
    cell["analysis"] = analyze(short, long)
    after = speed.scrape_metrics(args.base, args.api_key, args.timeout)
    cell["metrics_after"] = metric_snapshot(after)
    errors = cell["analysis"]["errors"]
    if any(after.get(name) != 0 for name in OCCUPANCY):
        errors.append("server is not idle or occupancy metrics are missing after the cell")
    for name, expected in ((speed.CACHE_HITS, 0), (SUCCESS, decoders + 1)):
        try:
            delta = counter_delta(before, after, name)
            cell.setdefault("counter_deltas", {})[name] = delta
            if delta != expected:
                errors.append(f"unexpected {name} delta: {delta}, expected {expected}")
        except ValueError as exc:
            errors.append(str(exc))
    spec_result = speed.acceptance(before, after)
    cell["acceptance"] = asdict(spec_result)
    if spec_result.error or spec_result.drafts <= 0 or spec_result.drafts > spec_result.drafted:
        errors.append(spec_result.error or "missing/inconsistent speculative drafts")
    cell["ok"] = not errors and launch
    cell["analysis"]["ok"] = cell["ok"]
    cell["complete"] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="nvidia/GLM-5.3-Flash-FP4-Local")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefill-lengths", default="16384,65536,131072")
    parser.add_argument("--decoders", default="1,3")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=2048)
    parser.add_argument("--max-seqs", type=int, default=4)
    parser.add_argument("--ready-tokens", type=int, default=32)
    parser.add_argument("--ready-timeout", type=float, default=120)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--seed", type=int, default=2026110306)
    args = parser.parse_args(argv)
    lengths = [int(v) for v in args.prefill_lengths.split(",")]
    decoders = [int(v) for v in args.decoders.split(",")]
    if (min(lengths + decoders) <= 0 or len(set(lengths)) != len(lengths)
            or len(set(decoders)) != len(decoders) or args.repetitions < 1
            or max(decoders) + 1 > args.max_seqs or not 1 <= args.ready_tokens < min(args.decode_tokens, 256)
            or not all(math.isfinite(v) and v > 0 for v in (args.timeout, args.ready_timeout))):
        parser.error("invalid grid, readiness budget or admission cap")
    if args.out.exists():
        parser.error("refusing to overwrite a receipt")
    args.api_key = os.environ.get("VLLM_API_KEY", "")
    receipt = {"schema": "mixed-prefill-v1", "complete": False, "ok": False,
               "started": datetime.now(timezone.utc).isoformat(),
               "controls": {k: v for k, v in vars(args).items() if k not in ("out", "api_key")},
               "warmup_prefill_tokens": 32768, "warmup_decode_tokens": 256,
               "decode_chat_template_kwargs": {"reasoning_effort": "max"},
               "decode_token_scope": "generated_tokens_including_reasoning",
               "measurement": "client cumulative-usage updates; no within-bundle token timing or GPU co-execution claim",
               "gap_definition": "full gap between consecutive usage updates intersecting long request start-to-first-update interval",
               "acceptance_scope": "counter deltas over the whole cell, including short decodes and the one-token long request",
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "speed_helper_sha256": hashlib.sha256(HELPER.read_bytes()).hexdigest(),
               "warmups": [], "cells": []}

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(args.out.name + ".tmp")
        temporary.write_text(json.dumps(receipt, default=asdict, indent=2) + "\n")
        os.replace(temporary, args.out)

    save()
    try:
        speed._request_json(args.base.rstrip("/") + "/health", None, args.api_key, 30)
        for warmup, rep, count, length, budget in (
            [(True, 0, c, 32768, 256) for c in decoders]
            + [(False, rep, c, length, args.decode_tokens)
               for rep in range(1, args.repetitions + 1) for c in decoders for length in lengths]
        ):
            cell = {}
            receipt["warmups" if warmup else "cells"].append(cell)
            measure_cell(args, rep, count, length, budget, cell)
            save()
            print(f"rep={rep} decoders={count} prefill={length} ok={cell['ok']}", flush=True)
            if not cell["ok"]:
                raise ValueError("cell failed; incomplete evidence retained")
        receipt["complete"] = receipt["ok"] = True
    except (Exception, KeyboardInterrupt) as exc:
        receipt["error"] = str(exc)
    finally:
        receipt["finished"] = datetime.now(timezone.utc).isoformat()
        save()
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
