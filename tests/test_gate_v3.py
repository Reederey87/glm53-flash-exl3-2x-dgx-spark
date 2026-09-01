#!/usr/bin/env python3
"""Gate v3 behavior: bypass, hold, deadline, aging escalation and the late-path log lines,
driven with a fake monotonic clock against the overlay's own helper text."""
from __future__ import annotations

import importlib.util
import io
import os
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "overlay" / "patch_scheduler_decode_floor.py"


class Req:
    def __init__(self, rid, num_tokens, num_prompt_tokens=None, computed=0):
        self.request_id = rid
        self.num_tokens = num_tokens
        self.num_prompt_tokens = num_tokens if num_prompt_tokens is None else num_prompt_tokens
        self.num_computed_tokens = computed


def _helper_ns(env: dict, clock: list):
    spec = importlib.util.spec_from_file_location("gate_overlay", OVERLAY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k in list(os.environ):
        if k.startswith("GLM53_MIXED_PREFILL_"):
            del os.environ[k]
    os.environ.update(env)
    ns = {"os": os, "__name__": "gate_ns"}
    exec(mod.HELPER, ns)
    real = time.monotonic
    time.monotonic = lambda: clock[0]
    return ns, real


def _decoding_peer():
    return Req("peer", 100, num_prompt_tokens=50, computed=90)   # computed >= prompt -> decoding


def test_bypass_hold_deadline_and_aging() -> None:
    clock = [1000.0]
    ns, real = _helper_ns({"GLM53_MIXED_PREFILL_ESCALATE_MS": "10000", "GLM53_MIXED_PREFILL_LATE_CAP_MAX": "1792"}, clock)
    try:
        gate = ns["_glm53_mixed_prefill_gate"]
        running = [_decoding_peer()]
        # warm follow-up: remainder <= one page -> no policy at all
        assert gate(running, Req("warm", 50_000, computed=47_000), 47_000) is None
        # cold read behind a decoding peer: held (0) before the deadline
        cold = Req("cold", 100_000)
        out = io.StringIO()
        with redirect_stdout(out):
            assert gate(running, cold, 0) == 0
            clock[0] += 1.0
            assert gate(running, cold, 0) == 0
            clock[0] += 0.6                      # 1.6 s waited -> late, flat cap
            assert gate(running, cold, 0) == 512
            assert gate(running, cold, 512) == 512
            clock[0] += 10.0                     # one escalation interval -> 1024
            assert gate(running, cold, 1024) == 1024
            clock[0] += 10.0                     # -> 2048 clamped to LATE_CAP_MAX 1792
            assert gate(running, cold, 2048) == 1792
            clock[0] += 50.0                     # stays at the ceiling
            assert gate(running, cold, 4096) == 1792
            # cap never exceeds the remainder
            assert gate(running, cold, 100_000 - 3_600) == 1792 or gate(running, cold, 100_000 - 3_600) == 3_600
            # last page: bypass, and the crawl-end line fires exactly once
            assert gate(running, cold, 100_000 - 3_000) is None
            assert gate(running, cold, 100_000 - 2_000) is None
        log = out.getvalue()
        assert log.count("late-admit req=cold") == 1, log
        assert "cap=512->1024" in log and "cap=1024->1792" in log, log
        assert log.count("late-done req=cold") == 1 and "final_cap=1792" in log, log
        # no peer decoding -> no policy
        assert gate([], Req("solo", 100_000), 0) is None
    finally:
        time.monotonic = real


def test_flat_v2_crawl_when_escalation_off() -> None:
    clock = [5.0]
    ns, real = _helper_ns({}, clock)   # defaults: ESCALATE_MS=0
    try:
        gate = ns["_glm53_mixed_prefill_gate"]
        running = [_decoding_peer()]
        cold = Req("cold2", 60_000)
        out = io.StringIO()
        with redirect_stdout(out):
            assert gate(running, cold, 0) == 0
            clock[0] += 2.0
            assert gate(running, cold, 0) == 512
            clock[0] += 120.0
            assert gate(running, cold, 30_000) == 512   # never escalates
        assert "late-escalate" not in out.getvalue()
    finally:
        time.monotonic = real


def test_wait_forever_when_max_wait_zero() -> None:
    clock = [0.0]
    ns, real = _helper_ns({"GLM53_MIXED_PREFILL_MAX_WAIT_MS": "0"}, clock)
    try:
        gate = ns["_glm53_mixed_prefill_gate"]
        cold = Req("cold3", 60_000)
        for _ in range(3):
            clock[0] += 100.0
            assert gate([_decoding_peer()], cold, 0) == 0
    finally:
        time.monotonic = real


def test_preemption_rollback_starts_a_new_crawl_episode() -> None:
    clock = [100.0]
    ns, real = _helper_ns({"GLM53_MIXED_PREFILL_ESCALATE_MS": "10000"}, clock)
    try:
        gate = ns["_glm53_mixed_prefill_gate"]
        running = [_decoding_peer()]
        cold = Req("pre", 80_000)
        out = io.StringIO()
        with redirect_stdout(out):
            assert gate(running, cold, 0) == 0
            clock[0] += 2.0
            assert gate(running, cold, 0) == 512
            clock[0] += 25.0
            assert gate(running, cold, 20_000) == 1792          # escalated
            # preempted: num_computed_tokens rolls back -> episode resets, age survives (still late)
            assert gate(running, cold, 0) == 512
            assert gate(running, cold, 512) == 512
        log = out.getvalue()
        assert log.count("late-admit req=pre") == 2, log
    finally:
        time.monotonic = real


class SlottedReq:
    __slots__ = ("request_id", "num_tokens", "num_prompt_tokens", "num_computed_tokens")   # no __dict__, no __weakref__

    def __init__(self, rid, n):
        self.request_id, self.num_tokens, self.num_prompt_tokens, self.num_computed_tokens = rid, n, n, 0


def test_fallback_when_request_rejects_attributes() -> None:
    clock = [0.0]
    ns, real = _helper_ns({}, clock)
    try:
        gate = ns["_glm53_mixed_prefill_gate"]
        running = [_decoding_peer()]
        cold = SlottedReq("slot", 60_000)
        out = io.StringIO()
        with redirect_stdout(out):
            assert gate(running, cold, 0) == 0
            clock[0] += 2.0
            assert gate(running, cold, 0) == 512            # age was kept in the fallback map
        log = out.getvalue()
        assert "bounded id-keyed state map" in log and log.count("late-admit req=slot") == 1, log
    finally:
        time.monotonic = real


if __name__ == "__main__":
    test_bypass_hold_deadline_and_aging()
    test_flat_v2_crawl_when_escalation_off()
    test_wait_forever_when_max_wait_zero()
    test_preemption_rollback_starts_a_new_crawl_episode()
    test_fallback_when_request_rejects_attributes()
    print("gate v3 OK")
