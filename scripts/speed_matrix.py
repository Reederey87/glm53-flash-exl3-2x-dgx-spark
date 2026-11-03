#!/usr/bin/env python3
"""Speed matrix for a running OpenAI-compatible vLLM server.

For each concurrency in ``--concurrencies``:

* Prefill, no cache: unique random token-ID prompts at 8k/16k/32k/64k/128k.
  TTFT is measured from request start to the first streamed token; prefill
  tok/s is prompt_tokens / TTFT. The aggregate column is total prompt tokens /
  wall time for the cell (includes queueing when concurrency exceeds
  --max-seqs).
* Decode: ``--decode-tokens`` completion tokens per request with
  ``ignore_eos``. Decode tok/s is (completion_tokens - 1) / (last token time -
  first token time); the aggregate column uses the cell wall time.
* Speculative acceptance: deltas of
  ``vllm:spec_decode_num_{accepted_tokens,draft_tokens,drafts}_total`` across
  the cell. Mean acceptance length = 1 + accepted / drafts; draft acceptance
  rate = accepted / drafted. Prefill cells with one output token usually have
  no drafts; the fields are null.

Three repetitions use distinct, deterministic prompts. Start each arm with
fresh server cache state and the same seed; nonzero prefix-hit deltas invalidate
no-cache prefill cells. Missing usage, incomplete streams and counter resets
also invalidate receipts. Pass ``--max-seqs`` to record the admission cap.

Usage:
  python3 scripts/speed_matrix.py --base http://127.0.0.1:8000 \
      --out local/speed-matrix.json

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
DRAFTED = "vllm:spec_decode_num_draft_tokens_total"
DRAFTS = "vllm:spec_decode_num_drafts_total"
CACHE_HITS = "vllm:prefix_cache_hits_total"


@dataclass
class RequestResult:
    ok: bool
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    total_s: float
    decode_s: float
    error: str = ""


@dataclass
class Acceptance:
    accepted: float = 0.0
    drafted: float = 0.0
    drafts: float = 0.0
    mean_acceptance_length: float | None = None
    draft_acceptance_rate: float | None = None
    error: str = ""


@dataclass
class Cell:
    kind: str
    concurrency: int
    length: int
    requests: int
    ok: int
    errors: int
    wall_s: float
    ttft_median_s: float | None = None
    ttft_p90_s: float | None = None
    prefill_tokens_per_s_median: float | None = None
    prefill_tokens_per_s_aggregate: float | None = None
    decode_tokens_per_s_median: float | None = None
    decode_tokens_per_s_aggregate: float | None = None
    completion_tokens_median: float | None = None
    acceptance: Acceptance = field(default_factory=Acceptance)
    mem_available_gib_before: float | None = None
    mem_available_gib_after: float | None = None
    note: str = ""
    repetition: int = 1
    prefix_cache_hits: float | None = None
    validation_errors: list[str] = field(default_factory=list)
    request_results: list[RequestResult] = field(default_factory=list)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * pct
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def make_prompt(length: int, rng: random.Random, vocab: int = 150000) -> list[int]:
    """Unique random token IDs; the whole prompt is random, not just a suffix."""
    if length <= 0:
        raise ValueError("prompt length must be positive")
    return [rng.randrange(1000, vocab) for _ in range(length)]


def _request_json(url: str, payload: dict[str, Any] | None, api_key: str, timeout: float) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
        return response.status, json.loads(raw) if raw else None


def stream_completion(
    base: str,
    model: str,
    prompt: list[int],
    max_tokens: int,
    api_key: str,
    timeout: float,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    start = time.perf_counter()
    first: float | None = None
    last: float | None = None
    done = False
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    done = True
                    break
                try:
                    event = json.loads(body)
                except json.JSONDecodeError:
                    return RequestResult(False, len(prompt), 0, 0.0, 0.0, 0.0, "malformed SSE JSON")
                if not isinstance(event, dict) or event.get("error"):
                    return RequestResult(False, len(prompt), 0, 0.0, 0.0, 0.0, "SSE error event")
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not isinstance(choices, list) or any(not isinstance(choice, dict) for choice in choices):
                    return RequestResult(False, len(prompt), 0, 0.0, 0.0, 0.0, "malformed SSE choices")
                for choice in choices:
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                    text = choice.get("text") or ""
                    if not text:
                        continue
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return RequestResult(False, len(prompt), 0, 0.0, 0.0, 0.0, str(exc))

    end = time.perf_counter()
    # A speculative SSE event can contain several tokens. Never substitute
    # event counts for server-reported usage, even on an incomplete stream.
    completion_tokens = (usage or {}).get("completion_tokens")
    prompt_tokens = (usage or {}).get("prompt_tokens")
    errors = []
    if not done or finish_reason != "length":
        errors.append("stream did not finish the fixed decode budget")
    if type(completion_tokens) is not int or completion_tokens != max_tokens:
        errors.append("missing or incorrect completion-token usage")
    if type(prompt_tokens) is not int or prompt_tokens != len(prompt):
        errors.append("missing or incorrect prompt-token usage")
    if first is None:
        errors.append("no streamed output")
    if errors:
        return RequestResult(
            False, len(prompt), completion_tokens if type(completion_tokens) is int else 0,
            max(0.0, (first or end) - start), end - start, 0.0, "; ".join(errors),
        )
    decode_s = max(0.0, (last or end) - first)
    return RequestResult(
        ok=True,
        prompt_tokens=len(prompt),
        completion_tokens=completion_tokens,
        ttft_s=max(0.0, first - start),
        total_s=max(0.0, end - start),
        decode_s=decode_s,
    )


def parse_metrics(text: str) -> dict[str, float]:
    counters: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            head, raw_value = line.rsplit(None, 1)
            value = float(raw_value)
        except ValueError:
            continue
        name = head.split("{", 1)[0]
        counters[name] = counters.get(name, 0.0) + value
    return counters


def scrape_metrics(base: str, api_key: str, timeout: float) -> dict[str, float]:
    headers = {"Accept": "text/plain"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{base.rstrip('/')}/metrics", headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return parse_metrics(response.read().decode("utf-8", errors="replace"))


def acceptance(before: dict[str, float], after: dict[str, float]) -> Acceptance:
    names = (ACCEPTED, DRAFTED, DRAFTS)
    if any(name not in before or name not in after for name in names):
        return Acceptance(error="missing speculative counters")
    if any(not math.isfinite(sample[name]) or sample[name] < 0 for sample in (before, after) for name in names):
        return Acceptance(error="invalid speculative counters")
    if any(after[name] < before[name] for name in names):
        return Acceptance(error="speculative counters reset during measurement")

    def delta(name: str) -> float:
        return after[name] - before[name]

    accepted = delta(ACCEPTED)
    drafted = delta(DRAFTED)
    drafts = delta(DRAFTS)
    if accepted > drafted or (drafts == 0 and (accepted or drafted)):
        return Acceptance(error="inconsistent speculative counter deltas")
    return Acceptance(
        accepted=accepted,
        drafted=drafted,
        drafts=drafts,
        mean_acceptance_length=(1.0 + accepted / drafts) if drafts else None,
        draft_acceptance_rate=(accepted / drafted) if drafted else None,
    )


def mem_available_gib() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576
    except OSError:
        return None
    return None


def run_cell(
    base: str,
    model: str,
    api_key: str,
    kind: str,
    concurrency: int,
    length: int,
    decode_tokens: int,
    timeout: float,
    seed: int,
    max_seqs: int,
    repetition: int = 1,
    expect_speculation: bool = True,
) -> Cell:
    requests = concurrency
    max_tokens = 1 if kind == "prefill" else decode_tokens
    prompts: list[list[int]] = []
    for index in range(requests):
        rng = random.Random(seed + concurrency * 100003 + length * 17 + index)
        prompts.append(make_prompt(length, rng))

    before_metrics = scrape_metrics(base, api_key, timeout)
    before_mem = mem_available_gib()
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(stream_completion, base, model, prompt, max_tokens, api_key, timeout)
            for prompt in prompts
        ]
        results = [future.result() for future in futures]
    wall_s = time.perf_counter() - wall_start
    after_metrics = scrape_metrics(base, api_key, timeout)
    after_mem = mem_available_gib()

    ok = [result for result in results if result.ok]
    errors = [result for result in results if not result.ok]
    ttfts = [result.ttft_s for result in ok]
    prefill_rates = [result.prompt_tokens / result.ttft_s for result in ok if result.ttft_s > 0]
    decode_rates = [
        (result.completion_tokens - 1) / result.decode_s
        for result in ok
        if result.decode_s > 0 and result.completion_tokens > 1
    ]
    completion_tokens = [result.completion_tokens for result in ok]
    note = ""
    if concurrency > max_seqs:
        note = f"concurrency {concurrency} > max_num_seqs {max_seqs}; queued"

    spec = acceptance(before_metrics, after_metrics) if expect_speculation else Acceptance()
    validation_errors = [spec.error] if spec.error else []
    hits = None
    if kind == "prefill":
        if CACHE_HITS not in before_metrics or CACHE_HITS not in after_metrics:
            validation_errors.append("missing prefix-cache hit counter")
        else:
            hits = after_metrics[CACHE_HITS] - before_metrics[CACHE_HITS]
            if not math.isfinite(hits) or hits != 0:
                validation_errors.append("no-cache prefill has cache hits or a counter reset")
    if kind == "decode" and expect_speculation and not spec.error and spec.drafts == 0:
        validation_errors.append("no speculative drafts observed during decode")

    cell = Cell(
        kind=kind,
        concurrency=concurrency,
        length=length,
        requests=requests,
        ok=len(ok),
        errors=len(errors),
        wall_s=wall_s,
        ttft_median_s=statistics.median(ttfts) if ttfts else None,
        ttft_p90_s=percentile(ttfts, 0.9),
        prefill_tokens_per_s_median=statistics.median(prefill_rates) if prefill_rates else None,
        prefill_tokens_per_s_aggregate=(sum(r.prompt_tokens for r in ok) / wall_s) if wall_s > 0 and ok else None,
        decode_tokens_per_s_median=statistics.median(decode_rates) if decode_rates else None,
        decode_tokens_per_s_aggregate=(
            sum(max(0, r.completion_tokens - 1) for r in ok) / wall_s
        ) if wall_s > 0 and ok and kind == "decode" else None,
        completion_tokens_median=statistics.median(completion_tokens) if completion_tokens else None,
        acceptance=spec,
        mem_available_gib_before=before_mem,
        mem_available_gib_after=after_mem,
        note=note,
        repetition=repetition,
        prefix_cache_hits=hits,
        validation_errors=validation_errors,
        request_results=results,
    )
    if errors:
        cell.note = (cell.note + "; " if cell.note else "") + f"first error: {errors[0].error[:120]}"
    return cell


def fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def markdown(cells: list[Cell], model: str, started: str) -> str:
    lines = [
        f"# Speed matrix — {model}",
        "",
        f"Started: {started}",
        "",
        "## Prefill, no cache (TTFT basis)",
        "",
        "| Rep | Conc | Prompt tokens | Req | TTFT med s | TTFT p90 s | Prefill tok/s med | Aggregate tok/s | Mem avail after GiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in [c for c in cells if c.kind == "prefill"]:
        lines.append(
            f"| {cell.repetition} | {cell.concurrency} | {cell.length:,} | {cell.requests} | {fmt(cell.ttft_median_s, 3)} | "
            f"{fmt(cell.ttft_p90_s, 3)} | {fmt(cell.prefill_tokens_per_s_median, 1)} | "
            f"{fmt(cell.prefill_tokens_per_s_aggregate, 1)} | {fmt(cell.mem_available_gib_after, 2)} |"
        )
    lines += [
        "",
        "## Decode",
        "",
        "| Rep | Conc | Prompt tokens | Completion med | Decode tok/s med | Aggregate tok/s | Mean accept len | Draft accept rate | Drafts | Mem avail after GiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in [c for c in cells if c.kind == "decode"]:
        lines.append(
            f"| {cell.repetition} | {cell.concurrency} | {cell.length} | {fmt(cell.completion_tokens_median, 0)} | "
            f"{fmt(cell.decode_tokens_per_s_median, 2)} | {fmt(cell.decode_tokens_per_s_aggregate, 2)} | "
            f"{fmt(cell.acceptance.mean_acceptance_length, 3)} | "
            f"{fmt(cell.acceptance.draft_acceptance_rate, 3)} | "
            f"{cell.acceptance.drafts:.0f} | {fmt(cell.mem_available_gib_after, 2)} |"
        )
    notes = [
        f"- rep={c.repetition} {c.kind} c={c.concurrency} len={c.length}: "
        + "; ".join(filter(None, [c.note, *c.validation_errors]))
        for c in cells if c.note or c.validation_errors
    ]
    if notes:
        lines += ["", "## Notes", ""] + notes
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="nvidia/GLM-5.3-Flash-FP4-Local")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--prefill-lengths", default="8192,16384,32768,65536,131072")
    parser.add_argument("--concurrencies", default="1,2,4,8")
    parser.add_argument("--decode-tokens", type=int, default=200)
    parser.add_argument("--max-seqs", type=int, default=4, help="server --max-num-seqs, for the note")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--no-speculation", action="store_true", help="only for a verified non-speculative control arm")
    parser.add_argument("--out", default="", help="JSON receipt")
    parser.add_argument("--markdown-out", default="", help="Markdown receipt")
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    args = parser.parse_args()

    prefill_lengths = [int(value) for value in args.prefill_lengths.split(",") if value.strip()]
    concurrencies = [int(value) for value in args.concurrencies.split(",") if value.strip()]
    if args.repetitions < 1 or args.decode_tokens < 2 or args.max_seqs < 1:
        parser.error("repetitions/max-seqs must be positive; decode-tokens must be >= 2")
    if not prefill_lengths or not concurrencies or min(prefill_lengths + concurrencies) < 1:
        parser.error("prefill-lengths and concurrencies must be nonempty positive lists")
    if args.skip_prefill and args.skip_decode:
        parser.error("cannot skip both prefill and decode")
    started = datetime.now(timezone.utc).isoformat()
    cells: list[Cell] = []

    try:
        status, _ = _request_json(f"{args.base.rstrip('/')}/health", None, args.api_key, 30)
        if status != 200:
            print(f"health returned HTTP {status}", file=sys.stderr)
            return 2
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"cannot reach {args.base}: {exc}", file=sys.stderr)
        return 2

    if args.warmup:
        print("warmup ...", file=sys.stderr)
        warm = stream_completion(args.base, args.model, make_prompt(512, random.Random(args.seed)), 16, args.api_key, args.timeout)
        if not warm.ok:
            print(f"warmup failed: {warm.error}", file=sys.stderr)
            return 2

    for repetition, concurrency in (
        (rep, conc) for rep in range(1, args.repetitions + 1) for conc in concurrencies
    ):
        # Identical across arms, distinct across reps. Start each arm with fresh
        # server state; the prefix-counter check rejects accidentally hot runs.
        cell_seed = args.seed + (repetition - 1) * 1000000007
        if not args.skip_prefill:
            for length in prefill_lengths:
                print(f"prefill c={concurrency} len={length} ...", file=sys.stderr)
                cell = run_cell(
                    args.base, args.model, args.api_key, "prefill", concurrency, length,
                    args.decode_tokens, args.timeout, cell_seed, args.max_seqs,
                    repetition, not args.no_speculation,
                )
                cells.append(cell)
                print(
                    f"  TTFT {fmt(cell.ttft_median_s, 3)}s, prefill {fmt(cell.prefill_tokens_per_s_median, 1)} tok/s, "
                    f"ok {cell.ok}/{cell.requests}",
                    file=sys.stderr,
                )
        if not args.skip_decode:
            print(f"decode c={concurrency} tokens={args.decode_tokens} ...", file=sys.stderr)
            cell = run_cell(
                args.base, args.model, args.api_key, "decode", concurrency, args.decode_tokens,
                args.decode_tokens, args.timeout, cell_seed, args.max_seqs,
                repetition, not args.no_speculation,
            )
            cells.append(cell)
            print(
                f"  decode {fmt(cell.decode_tokens_per_s_median, 2)} tok/s, "
                f"accept {fmt(cell.acceptance.mean_acceptance_length, 3)}, ok {cell.ok}/{cell.requests}",
                file=sys.stderr,
            )

    receipt = {
        "ok": bool(cells) and all(cell.errors == 0 and not cell.validation_errors for cell in cells),
        "base": args.base,
        "model": args.model,
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "prefill_lengths": prefill_lengths,
        "concurrencies": concurrencies,
        "decode_tokens": args.decode_tokens,
        "max_seqs": args.max_seqs,
        "seed": args.seed,
        "repetitions": args.repetitions,
        "expect_speculation": not args.no_speculation,
        "cells": [asdict(cell) for cell in cells],
    }
    print(markdown(cells, args.model, started))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
    if args.markdown_out:
        with open(args.markdown_out, "w", encoding="utf-8") as handle:
            handle.write(markdown(cells, args.model, started))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
