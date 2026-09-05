#!/usr/bin/env python3
"""Regression tests for the P0 streaming diagnostic."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "local" / "p0-runtime-probe.py"


def module():
    spec = importlib.util.spec_from_file_location("p0_runtime_probe", PROBE)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def base_run(**overrides):
    run = {
        "http": 200,
        "done": True,
        "sse_chunks": 3,
        "finish_reason": "stop",
        "usage": {"completion_tokens": 1200},
        "content_head": "THINKING_STREAM_OK",
        "content_tail": "",
        "diagnostics": {
            "replacement_or_mojibake": [],
            "max_16gram_repeats": 1,
            "content_chars": 18,
            "reasoning_chars": 20,
        },
    }
    run.update(overrides)
    return run


def test_thinking_stream_requires_final_answer() -> None:
    probe = module()
    run = base_run(
        finish_reason="length",
        content_head="",
        diagnostics={
            "replacement_or_mojibake": [],
            "max_16gram_repeats": 1,
            "content_chars": 0,
            "reasoning_chars": 400,
        },
    )
    failures = probe.validate_run("thinking-sse", run, 1000)
    assert "thinking-sse: finish_reason='length'" in failures
    assert "thinking-sse: missing final answer marker" in failures


def test_thinking_stream_accepts_reasoning_then_answer() -> None:
    probe = module()
    assert probe.validate_run("thinking-sse", base_run(), 1000) == []


class Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        events = (
            {"choices": [{"delta": {"content": "prefix\u0000suffix"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 2},
            },
        )
        for event in events:
            yield f"data: {json.dumps(event)}\n".encode()
        yield b"data: [DONE]\n"


def test_json_escaped_nul_is_detected(monkeypatch) -> None:
    probe = module()
    monkeypatch.setattr(probe, "open_url", lambda *_args, **_kwargs: Response())
    run = probe.stream_request(
        "http://example.invalid",
        "prompt",
        thinking=False,
        temperature=0.0,
        max_tokens=4,
        timeout=1,
        request_id="test",
    )
    assert "\x00" in run["diagnostics"]["replacement_or_mojibake"]
