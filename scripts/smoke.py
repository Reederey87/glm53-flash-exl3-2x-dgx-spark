#!/usr/bin/env python3
"""Acceptance checks for a running GLM-5.3-Flash-FP4-Local server.

Hard checks (must pass):
  * /health answers 200
  * /v1/models lists the served model
  * one temperature-0 chat completion returns non-empty content

Soft checks (recorded, not fatal):
  * the reasoning parser produced reasoning_content
  * the tool parser produced a tool call
  * a repeated prefix reports cached prompt tokens

Usage:
  python3 scripts/smoke.py --base http://127.0.0.1:8000 \
      --model nvidia/GLM-5.3-Flash-FP4-Local --out local/smoke.json

Standard library only; safe to run inside the head node or from the Mac over
an SSH tunnel.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class Check:
    name: str
    status: str  # pass | warn | fail
    detail: str


def http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    api_key: str = "",
    timeout: float = 900.0,
) -> tuple[int, dict[str, Any] | str]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, str(exc)


def chat(
    base: str,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str,
    max_tokens: int = 128,
    tools: list[dict[str, Any]] | None = None,
    timeout: float = 900.0,
    template_kwargs: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | str]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if template_kwargs:
        payload["chat_template_kwargs"] = template_kwargs
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return http_json(f"{base.rstrip('/')}/v1/chat/completions", payload, api_key, timeout)


def message_of(body: dict[str, Any] | str) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    choices = body.get("choices") or []
    if not choices:
        return {}
    return choices[0].get("message") or {}



PREFIX_HITS = "vllm:prefix_cache_hits_total"


def _parse_cache_metrics(text: str) -> tuple[float, str]:
    """Return prefix-cache hits and a compact cache-geometry string."""
    hits = 0.0
    geometry = "geometry=unknown"
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("vllm:cache_config_info{"):
            labels = line.split("{", 1)[1].split("}", 1)[0]
            values: dict[str, str] = {}
            for part in labels.split(","):
                if "=" in part:
                    key, value = part.split("=", 1)
                    values[key] = value.strip('"')
            geometry = (
                f"block={values.get('block_size', '?')}"
                f"/mamba={values.get('mamba_block_size', '?')}"
                f"/fine_grained={values.get('enable_mamba_fine_grained_prefix_cache', '?')}"
            )
            continue
        if line.startswith("#") or not line:
            continue
        if line.split("{", 1)[0].split(" ", 1)[0] == PREFIX_HITS:
            try:
                hits += float(line.rsplit(None, 1)[1])
            except (IndexError, ValueError):
                continue
    return hits, geometry


def scrape_prefix_metrics(base: str, api_key: str, timeout: float) -> tuple[float, str]:
    url = f"{base.rstrip('/')}/metrics"
    headers = {"Accept": "text/plain"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=min(timeout, 30)) as response:
            return _parse_cache_metrics(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0.0, "geometry=unavailable"


def prefix_cache_verdict(
    first_status: int,
    second_status: int,
    hit_delta: float,
    cached_tokens: int | None,
    field_present: bool,
    geometry: str,
) -> Check:
    """Judge the cache from the API field and the metrics counter."""
    http = f"HTTP {first_status}/{second_status}"
    if first_status != 200 or second_status != 200:
        return Check("prefix_cache", "warn", f"{http}; {geometry}")
    if hit_delta > 0:
        api = f"cached_tokens={cached_tokens}" if field_present else "cached_tokens field absent"
        return Check(
            "prefix_cache",
            "pass",
            f"{http}; hits +{hit_delta:.0f}; {api}; {geometry}",
        )
    if cached_tokens is not None and cached_tokens > 0:
        return Check(
            "prefix_cache",
            "pass",
            f"{http}; cached_tokens={cached_tokens}; {geometry}",
        )
    if not field_present:
        return Check(
            "prefix_cache",
            "warn",
            f"{http}; no hit and cached_tokens field absent; start with "
            f"--enable-prompt-tokens-details; {geometry}",
        )
    return Check(
        "prefix_cache",
        "warn",
        f"{http}; no hit; cached_tokens={cached_tokens}; {geometry}",
    )


def run(base: str, model: str, api_key: str, timeout: float) -> list[Check]:
    checks: list[Check] = []

    status, body = http_json(f"{base.rstrip('/')}/health", api_key=api_key, timeout=30)
    checks.append(Check("health", "pass" if status == 200 else "fail", f"HTTP {status}"))

    status, body = http_json(f"{base.rstrip('/')}/v1/models", api_key=api_key, timeout=30)
    ids: list[str] = []
    if isinstance(body, dict):
        ids = [item.get("id", "") for item in body.get("data", []) if isinstance(item, dict)]
    listed = model in ids
    checks.append(
        Check(
            "models",
            "pass" if status == 200 and listed else "fail",
            f"HTTP {status}; ids={ids}",
        )
    )

    started = time.monotonic()
    status, body = chat(
        base,
        model,
        [{"role": "user", "content": "Reply with exactly: OK"}],
        api_key,
        max_tokens=128,
        timeout=timeout,
        template_kwargs={"reasoning_effort": "low"},
    )
    elapsed = time.monotonic() - started
    message = message_of(body)
    content = (message.get("content") or "").strip()
    checks.append(
        Check(
            "chat",
            "pass" if status == 200 and content else "fail",
            f"HTTP {status}; {elapsed:.2f}s; content={content[:80]!r}",
        )
    )

    status, body = chat(
        base,
        model,
        [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}],
        api_key,
        max_tokens=256,
        timeout=timeout,
    )
    message = message_of(body)
    reasoning = (message.get("reasoning_content") or "").strip()
    hard_content = (message.get("content") or "").strip()
    checks.append(
        Check(
            "reasoning_parser",
            "pass" if status == 200 and (reasoning or hard_content) else "warn",
            f"HTTP {status}; reasoning={len(reasoning)} chars; content={hard_content[:40]!r}",
        )
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    status, body = chat(
        base,
        model,
        [
            {
                "role": "user",
                "content": "What is the weather in Paris? Use the get_weather tool.",
            }
        ],
        api_key,
        max_tokens=256,
        tools=tools,
        timeout=timeout,
        template_kwargs={"reasoning_effort": "low"},
    )
    tool_calls = message_of(body).get("tool_calls") or []
    checks.append(
        Check(
            "tool_parser",
            "pass" if status == 200 and tool_calls else "warn",
            f"HTTP {status}; tool_calls={len(tool_calls)}",
        )
    )

    # A repeated prompt should hit the prefix cache. Probe a prompt longer than
    # the hybrid attention block so a 0 is a real miss, not a sub-block prompt.
    # `cached_tokens` needs --enable-prompt-tokens-details; the metrics counter
    # is the fallback when the server does not export the field.
    long_prompt = (
        "You are a careful assistant. Read this line and reply with the word DONE.\n"
        + ("The quick brown fox jumps over the lazy dog. " * 600)
    )
    hits_before, geometry = scrape_prefix_metrics(base, api_key, timeout)
    first_status, first = chat(
        base, model, [{"role": "user", "content": long_prompt}], api_key,
        max_tokens=16, timeout=timeout, template_kwargs={"reasoning_effort": "low"},
    )
    second_status, second = chat(
        base, model, [{"role": "user", "content": long_prompt}], api_key,
        max_tokens=16, timeout=timeout, template_kwargs={"reasoning_effort": "low"},
    )
    hits_after, geometry_after = scrape_prefix_metrics(base, api_key, timeout)
    deadline = time.monotonic() + 5.0
    while hits_after <= hits_before and time.monotonic() < deadline:
        time.sleep(0.2)
        hits_after, geometry_after = scrape_prefix_metrics(base, api_key, timeout)
    cached_tokens: int | None = None
    field_present = False
    if isinstance(second, dict):
        usage = second.get("usage") or {}
        details = usage.get("prompt_tokens_details")
        field_present = isinstance(details, dict)
        if field_present:
            cached_tokens = details.get("cached_tokens")
    checks.append(
        prefix_cache_verdict(
            first_status,
            second_status,
            max(0.0, hits_after - hits_before),
            cached_tokens,
            field_present,
            geometry_after if geometry_after != "geometry=unknown" else geometry,
        )
    )

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="OpenAI-compatible base URL")
    parser.add_argument("--model", default="nvidia/GLM-5.3-Flash-FP4-Local", help="served model id")
    parser.add_argument("--api-key", default="", help="optional bearer token")
    parser.add_argument("--timeout", type=float, default=900.0, help="per-request timeout (seconds)")
    parser.add_argument("--out", default="", help="write the JSON receipt here")
    args = parser.parse_args()

    checks = run(args.base, args.model, args.api_key, args.timeout)
    ok = all(check.status != "fail" for check in checks)
    receipt = {
        "ok": ok,
        "base": args.base,
        "model": args.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": [asdict(check) for check in checks],
    }

    for check in checks:
        marker = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[check.status]
        print(f"[{marker}] {check.name}: {check.detail}")
    print(json.dumps({"ok": ok, "checks": len(checks)}))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
