#!/usr/bin/env python3
"""P0 long-form corruption and thinking-SSE probe.

Runs three fixed English essay prompts with thinking disabled and one short
thinking-enabled streaming request. Every request uses APC no-store. The probe
fails on missing SSE termination, short/incomplete output, invalid UTF-8,
replacement/mojibake markers, repeated-token loops, HTTP errors, preemptions, or
requests left running/waiting.

Run from the Mac through the production tunnel:

  uv run python local/p0-runtime-probe.py \
    --base http://127.0.0.1:18000 \
    --out local/p0-runtime-$(date +%F).json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


MODEL = os.environ.get("GLM53_MODEL", "GLM-5.3-Flash-EXL3")
API_KEY = os.environ.get("VLLM_API_KEY", "")
PROMPTS = (
    "Write a detailed essay explaining how ocean tides work, including lunar and solar forcing, spring and neap tides, amphidromic systems, coastal geometry, and common misconceptions. Use clear sections and concrete examples.",
    "Write a detailed essay on how a modern operating system manages virtual memory. Cover page tables, translation caches, page faults, copy-on-write, file-backed mappings, swapping, and failure modes under memory pressure.",
    "Write a detailed essay explaining the scientific method in practice. Cover hypothesis formation, experimental design, measurement uncertainty, replication, falsification, peer review, and how scientists revise models after conflicting evidence.",
)
MOJIBAKE = ("�", "Ã", "Â", "â€", "\x00")
METRICS = (
    "num_requests_running",
    "num_requests_waiting",
    "num_preemptions_total",
    "prefix_cache_queries_total",
    "prefix_cache_hits_total",
)


def headers(request_id: str = "") -> dict[str, str]:
    out = {"Content-Type": "application/json"}
    if API_KEY:
        out["Authorization"] = f"Bearer {API_KEY}"
    if request_id:
        out["X-Request-Id"] = request_id
    return out


def open_url(base: str, path: str, body: dict | None, timeout: float, request_id: str = ""):
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers=headers(request_id),
        method="POST" if body is not None else "GET",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def metrics(base: str) -> dict[str, float]:
    with open_url(base, "/metrics", None, 30) as response:
        text = response.read().decode("utf-8", "strict")
    result: dict[str, float] = {}
    for name in METRICS:
        values = re.findall(
            rf"^vllm:{re.escape(name)}(?:\{{[^}}]*\}})?\s+(\S+)$", text, re.MULTILINE
        )
        if values:
            result[name] = sum(float(value) for value in values)
    return result


def max_ngram_repeats(text: str, width: int = 16) -> int:
    words = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(words) < width:
        return 0
    counts = Counter(tuple(words[i : i + width]) for i in range(len(words) - width + 1))
    return max(counts.values(), default=0)


def stream_request(
    base: str,
    prompt: str,
    *,
    thinking: bool,
    temperature: float,
    max_tokens: int,
    timeout: float,
    request_id: str,
) -> dict[str, Any]:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {
            "enable_thinking": thinking,
            "reasoning_effort": "max",
        },
        "vllm_xargs": {"skip_writing_prefix_cache": 1},
    }
    started = time.perf_counter()
    first_byte = None
    chunks = 0
    done = False
    usage: dict[str, Any] = {}
    finish_reason = None
    content: list[str] = []
    reasoning: list[str] = []
    with open_url(
        base, "/v1/chat/completions", body, timeout, request_id=request_id
    ) as response:
        status = response.status
        for raw in response:
            if first_byte is None:
                first_byte = time.perf_counter()
            line = raw.decode("utf-8", "strict").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                continue
            event = json.loads(payload)
            chunks += 1
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                value = delta.get("reasoning") or delta.get("reasoning_content")
                if value:
                    reasoning.append(value)
    ended = time.perf_counter()
    text = "".join(content)
    thought = "".join(reasoning)
    diagnostics = {
        "replacement_or_mojibake": [marker for marker in MOJIBAKE if marker in text],
        "max_16gram_repeats": max_ngram_repeats(text),
        "content_chars": len(text),
        "reasoning_chars": len(thought),
    }
    return {
        "request_id": request_id,
        "http": status,
        "sse_chunks": chunks,
        "done": done,
        "ttfb_s": round(first_byte - started, 3) if first_byte else None,
        "wall_s": round(ended - started, 3),
        "finish_reason": finish_reason,
        "usage": usage,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "content_head": text[:240],
        "content_tail": text[-240:],
        "diagnostics": diagnostics,
    }


def validate_run(name: str, run: dict[str, Any], min_essay_tokens: int) -> list[str]:
    failures: list[str] = []
    if run.get("error"):
        return [f"{name}: {run['error']}"]
    if run.get("http") != 200 or not run.get("done") or run.get("sse_chunks", 0) < 2:
        failures.append(f"{name}: incomplete SSE response")
    diagnostics = run["diagnostics"]
    if diagnostics["replacement_or_mojibake"]:
        failures.append(f"{name}: mojibake markers {diagnostics['replacement_or_mojibake']}")
    if diagnostics["max_16gram_repeats"] > 6:
        failures.append(f"{name}: repeated 16-gram loop")

    if name.startswith("essay"):
        if run.get("finish_reason") not in {"stop", "length"}:
            failures.append(f"{name}: finish_reason={run.get('finish_reason')!r}")
        if int(run.get("usage", {}).get("completion_tokens") or 0) < min_essay_tokens:
            failures.append(f"{name}: fewer than {min_essay_tokens} completion tokens")
    elif name == "thinking-sse":
        if run.get("finish_reason") != "stop":
            failures.append(f"{name}: finish_reason={run.get('finish_reason')!r}")
        if diagnostics["reasoning_chars"] == 0:
            failures.append("thinking-sse: no reasoning deltas")
        if "THINKING_STREAM_OK" not in (
            str(run.get("content_head", "")) + str(run.get("content_tail", ""))
        ):
            failures.append("thinking-sse: missing final answer marker")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=os.environ.get("GLM53_BASE", "http://127.0.0.1:18000"))
    parser.add_argument("--essay-max-tokens", type=int, default=1800)
    parser.add_argument("--thinking-max-tokens", type=int, default=256)
    parser.add_argument("--min-essay-tokens", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    before = metrics(args.base)
    if not args.force and (
        before.get("num_requests_running", 0) > 0
        or before.get("num_requests_waiting", 0) > 0
    ):
        print("server busy; refusing (use --force only for an intentional overlap)")
        return 2

    run_id = uuid.uuid4().hex[:10]
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "started": time.time(),
        "base": args.base,
        "model": MODEL,
        "before": before,
        "runs": [],
    }
    failures: list[str] = []
    cells = [
        (f"essay-{index + 1}", prompt, False, 0.7, args.essay_max_tokens)
        for index, prompt in enumerate(PROMPTS)
    ]
    cells.append(
        (
            "thinking-sse",
            "Think briefly, then answer with exactly: THINKING_STREAM_OK",
            True,
            0.0,
            args.thinking_max_tokens,
        )
    )

    for name, prompt, thinking, temperature, max_tokens in cells:
        request_id = f"glm53-p0-{run_id}-{name}"
        try:
            run = stream_request(
                args.base,
                prompt,
                thinking=thinking,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=args.timeout,
                request_id=request_id,
            )
        except (urllib.error.URLError, OSError, TimeoutError, UnicodeError, json.JSONDecodeError) as exc:
            run = {"request_id": request_id, "error": f"{type(exc).__name__}: {exc}"}
        run["name"] = name
        run["thinking"] = thinking
        result["runs"].append(run)
        failures.extend(validate_run(name, run, args.min_essay_tokens))

    after = metrics(args.base)
    result["after"] = after
    result["metric_delta"] = {
        key: after.get(key, 0) - before.get(key, 0) for key in METRICS
    }
    if result["metric_delta"].get("num_preemptions_total", 0) != 0:
        failures.append("preemption counter increased")
    if after.get("num_requests_running", 0) or after.get("num_requests_waiting", 0):
        failures.append("requests remain running or waiting")
    result["failures"] = failures
    result["passed"] = not failures
    result["finished"] = time.time()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "failures": failures, "out": str(out)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
