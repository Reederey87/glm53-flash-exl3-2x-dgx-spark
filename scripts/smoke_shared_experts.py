#!/usr/bin/env python3
"""Tool-serving smoke for a shared-expert-launch arm.

The pre-registered gate in ``docs/21-shared-experts-overlap.md`` requires, before
adopting: four concurrent requests plus plain completions, with correct tool
arguments and no engine, CUDA, NCCL or NaN errors.

This fires a mixed concurrent batch — two tool-calling requests that must return
parseable arguments for the declared schema, and two plain completions — then
scans the head and worker container logs for engine-level errors.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "http://127.0.0.1:8000"
MODEL = "GLM-5.3-Flash-EXL3"
HEAD_CONTAINER = "glm53-exl3-head"
WORKER_CONTAINER = "glm53-exl3-worker"
WORKER = "nvidia@192.168.177.11"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "units"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two integers and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    },
]

NAN_RE = re.compile(r"\bnan\b|locklock", re.I)
ERROR_RE = re.compile(
    r"CUDA error|illegal memory access|NCCL error|NCCL WARN|RuntimeError|"
    r"EngineDeadError|engine core.*died|Traceback \(most recent call last\)",
    re.I,
)

REQUESTS = [
    ("tool-weather", "What is the weather in Lisbon right now? Use the tool, in celsius.", True),
    ("tool-add", "Add 4217 and 1938. Use the tool.", True),
    ("plain-1", "Reply with exactly the word READY.", False),
    ("plain-2", "Name the three primary colours, comma separated. No other text.", False),
]


def _post(payload: dict, timeout: int = 180) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _one(name: str, prompt: str, wants_tool: bool) -> dict:
    payload: dict = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
    }
    if wants_tool:
        payload["tools"] = TOOLS
        payload["tool_choice"] = "auto"
    started = time.time()
    record: dict = {"name": name, "wants_tool": wants_tool, "started": started}
    try:
        data = _post(payload)
    except urllib.error.HTTPError as exc:
        record.update(ok=False, http=exc.code, error=exc.read().decode()[:500])
        return record
    except Exception as exc:  # noqa: BLE001
        record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        return record
    record["elapsed_s"] = time.time() - started
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    calls = message.get("tool_calls") or []
    record["finish_reason"] = choice.get("finish_reason")
    record["content_head"] = content[:200]
    record["n_tool_calls"] = len(calls)
    record["nan"] = bool(NAN_RE.search(content))

    args_ok = True
    arg_detail = []
    for call in calls:
        fn = (call.get("function") or {})
        raw = fn.get("arguments") or ""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            args_ok = False
            arg_detail.append(f"{fn.get('name')}: unparseable arguments ({exc})")
            continue
        if not isinstance(parsed, dict):
            args_ok = False
            arg_detail.append(f"{fn.get('name')}: arguments are not an object")
            continue
        if fn.get("name") == "get_weather":
            good = isinstance(parsed.get("city"), str) and parsed.get("units") in (
                "celsius",
                "fahrenheit",
            )
            args_ok &= good
            arg_detail.append(f"get_weather args={parsed} ok={good}")
        elif fn.get("name") == "add_numbers":
            good = isinstance(parsed.get("a"), int) and isinstance(parsed.get("b"), int)
            args_ok &= good
            arg_detail.append(f"add_numbers args={parsed} ok={good}")
    record["tool_args_ok"] = args_ok if wants_tool else None
    record["tool_arg_detail"] = arg_detail
    record["tool_call_required"] = wants_tool
    if wants_tool:
        record["ok"] = bool(calls) and args_ok and not record["nan"]
    else:
        record["ok"] = bool(content.strip()) and not record["nan"]
    return record


def _logs(container: str, host: str, since: float) -> str:
    argv = ["docker", "logs", "--since", str(int(since)), container]
    if host != "local":
        argv = ["ssh", "-o", "ConnectTimeout=10", host] + argv
    done = subprocess.run(argv, capture_output=True, text=True, timeout=180, check=False)
    return done.stdout + done.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="local/shared-experts-20260927/smoke.json")
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    started = time.time() - 5
    results: list[dict] = []
    for round_no in range(args.repeat):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_one, *spec) for spec in REQUESTS]
            batch = [f.result() for f in futures]
        for row in batch:
            row["round"] = round_no + 1
        results.extend(batch)
        print(
            json.dumps(
                [
                    {
                        "name": r["name"],
                        "round": r["round"],
                        "ok": r.get("ok"),
                        "n_tool_calls": r.get("n_tool_calls"),
                        "tool_args_ok": r.get("tool_args_ok"),
                        "nan": r.get("nan"),
                    }
                    for r in batch
                ]
            ),
            flush=True,
        )

    head_log = _logs(HEAD_CONTAINER, "local", started)
    worker_log = _logs(WORKER_CONTAINER, WORKER, started)
    log_errors = {
        "head": sorted(set(ERROR_RE.findall(head_log))),
        "worker": sorted(set(ERROR_RE.findall(worker_log))),
    }

    bad = [r for r in results if not r.get("ok")]
    receipt = {
        "ts": time.time(),
        "repeat": args.repeat,
        "requests": len(results),
        "failed": len(bad),
        "failed_names": [f"{r['name']}#{r.get('round')}" for r in bad],
        "any_nan": any(r.get("nan") for r in results),
        "log_errors": log_errors,
        "ok": not bad and not any(log_errors.values()),
        "results": results,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k != "results"}, indent=2))
    print("wrote", path)
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
