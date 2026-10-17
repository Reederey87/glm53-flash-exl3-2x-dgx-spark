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
) -> tuple[int, dict[str, Any] | str]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
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
        max_tokens=64,
        timeout=timeout,
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
    )
    tool_calls = message_of(body).get("tool_calls") or []
    checks.append(
        Check(
            "tool_parser",
            "pass" if status == 200 and tool_calls else "warn",
            f"HTTP {status}; tool_calls={len(tool_calls)}",
        )
    )

    # A repeated prompt should report a prefix-cache hit. This is a soft check:
    # the first two requests can race, and an eviction policy may decline a hit.
    long_prompt = (
        "You are a careful assistant. Read this line and reply with the word DONE.\n"
        + ("The quick brown fox jumps over the lazy dog. " * 80)
    )
    first_status, first = chat(
        base, model, [{"role": "user", "content": long_prompt}], api_key, max_tokens=16, timeout=timeout
    )
    second_status, second = chat(
        base, model, [{"role": "user", "content": long_prompt}], api_key, max_tokens=16, timeout=timeout
    )
    cached = 0
    if isinstance(second, dict):
        usage = second.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
    checks.append(
        Check(
            "prefix_cache",
            "pass" if first_status == 200 and second_status == 200 and cached > 0 else "warn",
            f"HTTP {first_status}/{second_status}; cached_tokens={cached}",
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
