"""CPU-only tests for the task 35 §6 qualification window.

No cluster, no torch, no HTTP: the probe's pure helpers, the auditor's
decision logic, and the runner's arm/phase wiring are all exercised against
synthetic receipts. The real window's observations come from the cluster; these
tests exist so a parser or contract edit cannot silently change a verdict.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

KIT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = KIT_ROOT / "scripts"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load(SCRIPTS / "probe_v149_qualification.py", "glm53_probe_v149")
audit = _load(SCRIPTS / "audit_v149_qualification.py", "glm53_audit_v149")
window = _load(SCRIPTS / "run_v149_qualification_window.py", "glm53_window_v149")


# --- probe ------------------------------------------------------------------

def test_probe_kinds_cover_every_contract_lane():
    assert set(probe.KINDS) == set(audit.LANES)


def test_probe_structured_prompt_is_the_standing_gate_payload():
    # Same payload as tests/bench_decode.py's STRUCTURED_PROMPT; the whole point
    # is comparability with the 69-70 tok/s band in the existing receipts.
    assert probe.STRUCTURED_PROMPT.startswith("Count from 1 to 200.")


def _decode_run(tok_s=70.0, completion_tokens=200, nan=False, ttft_s=0.3, http=200, error=None):
    return {
        "http": http,
        "error": error,
        "ttft_s": ttft_s,
        "decode_s": 2.0,
        "tok_s": tok_s,
        "completion_tokens": completion_tokens,
        "prompt_tokens": 34,
        "finish_reason": "length",
        "nan": nan,
        "text_head": "1 2 3",
        "spec": {"drafts": 25, "draft_tokens": 175, "accepted": 175,
                 "accept_ratio": 1.0, "accepted_per_step": 7.0, "pos": [1.0] * 7},
    }


def _prefill_run(rate=1400.0, prompt_tokens=None, cached_tokens=0, target=60_000, ttft_s=57.0):
    prompt_tokens = target if prompt_tokens is None else prompt_tokens
    return {
        "http": 200,
        "error": None,
        "target": target,
        "filler_count": 10,
        "tokenize_estimate": prompt_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "ttft_s": ttft_s,
        "wall_s": ttft_s + 1,
        "prefill_tok_s": rate,
        "finish_reason": "stop",
        "nan": False,
        "text_head": "OK",
    }


def test_decode_run_invalid_accepts_a_full_block():
    assert probe.decode_run_invalid(_decode_run(), 200) is None


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"nan": True}, "NaN"),
        ({"http": 500}, "http 500"),
        ({"ttft_s": None}, "no first token"),
        ({"completion_tokens": 12}, "short completion"),
        ({"tok_s": None}, "completion_tokens"),
        ({"error": "URLError: refused"}, "URLError"),
    ],
)
def test_decode_run_invalid_rejects(kwargs, needle):
    run = _decode_run()
    run.update(kwargs)
    reason = probe.decode_run_invalid(run, 200)
    assert reason is not None and needle in reason


def test_prefill_run_invalid_accepts_a_cold_run():
    assert probe.prefill_run_invalid(_prefill_run()) is None


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"cached_tokens": 4096}, "warm request"),
        ({"cached_tokens": None}, "cannot prove the run was cold"),
        ({"prompt_tokens": 20_000}, "outside"),
        ({"http": 503}, "http 503"),
        ({"ttft_s": None}, "no first token"),
        ({"prefill_tok_s": None}, "no prefill rate"),
        ({"prompt_tokens": 0}, "no prompt_tokens"),
    ],
)
def test_prefill_run_invalid_rejects(kwargs, needle):
    run = _prefill_run()
    run.update(kwargs)
    reason = probe.prefill_run_invalid(run)
    assert reason is not None and needle in reason


def test_summarize_reports_median_and_flags_nan_and_cache_hits():
    decode = probe.summarize("structured", [_decode_run(70.0), _decode_run(80.0)], [])
    assert decode["valid_runs"] == 2
    assert decode["tok_s_median"] == 75.0
    assert decode["any_nan"] is False
    assert decode["accepted_per_step_median"] == 7.0
    flagged = probe.summarize("structured", [_decode_run(nan=True)], [])
    assert flagged["any_nan"] is True
    prefill = probe.summarize("prefill60k", [_prefill_run(cached_tokens=99)], [])
    assert prefill["any_cache_hit"] is True
    assert prefill["prefill_tok_s_median"] == 1400.0


def test_probe_refuses_to_measure_an_unhealthy_server(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "health", lambda: 503)
    rc = probe.main(["--kind", "structured", "--runs", "1", "--out", str(tmp_path / "o.json")])
    assert rc == 2
    assert not (tmp_path / "o.json").exists()


def test_probe_exits_nonzero_when_a_run_is_invalid(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "health", lambda: 200)
    monkeypatch.setattr(probe, "served_model", lambda: "GLM-5.3-Flash-EXL3")
    monkeypatch.setattr(probe, "spec_snapshot", lambda: {})
    monkeypatch.setattr(probe, "decode_run", lambda *a, **k: _decode_run(nan=True))
    out = tmp_path / "o.json"
    rc = probe.main(["--kind", "structured", "--runs", "3", "--out", str(out)])
    assert rc == 1
    payload = json.loads(out.read_text())
    assert payload["valid_runs"] == 0 and len(payload["invalid_runs"]) == 3


def test_probe_keeps_completed_observations_when_a_later_run_fails(monkeypatch, tmp_path):
    """A crash mid-block must not discard the observations already taken."""
    monkeypatch.setattr(probe, "health", lambda: 200)
    monkeypatch.setattr(probe, "served_model", lambda: "GLM-5.3-Flash-EXL3")
    monkeypatch.setattr(probe, "spec_snapshot", lambda: {})
    calls = {"n": 0}

    def flaky(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("connection reset")
        return _decode_run(tok_s=70.0 + calls["n"])

    monkeypatch.setattr(probe, "decode_run", flaky)
    out = tmp_path / "o.json"
    with pytest.raises(RuntimeError, match="connection reset"):
        probe.main(["--kind", "structured", "--runs", "5", "--out", str(out)])
    payload = json.loads(out.read_text())
    assert payload["valid_runs"] == 2
    assert [r["tok_s"] for r in payload["runs"]] == [71.0, 72.0]


def test_probe_receipt_write_is_atomic(monkeypatch, tmp_path):
    out = tmp_path / "o.json"
    probe.write_receipt(out, {"a": 1})
    assert json.loads(out.read_text()) == {"a": 1}
    assert not (tmp_path / "o.json.tmp").exists()


class _FakeStream:
    """A minimal SSE response that records how it was read."""

    def __init__(self, lines, *, status=200, delay=0.0):
        self._lines = list(lines)
        self.status = status
        self._delay = delay
        self.read_calls = 0

    def readline(self):
        if not self._lines:
            return b""
        if self._delay:
            time.sleep(self._delay)
        return self._lines.pop(0)

    def read(self, *_a):  # pragma: no cover - must never be reached
        self.read_calls += 1
        raise AssertionError("the stream must be read line-by-line, not buffered")

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _sse(*contents, usage=None):
    lines = [
        b'data: {"choices":[{"delta":{"content":"' + c.encode() + b'"}}]}\n'
        for c in contents
    ]
    if usage:
        lines.append(
            b'data: {"choices":[],"usage":' + json.dumps(usage).encode() + b"}\n"
        )
    lines.append(b"data: [DONE]\n")
    return lines


def test_stream_reads_line_by_line_not_in_buffered_blocks(monkeypatch):
    """`read(4096)` blocks until 4096 bytes accumulate, folding several token
    arrivals into one and collapsing the measured decode interval."""
    fake = _FakeStream(_sse("a", "b"))
    monkeypatch.setattr(probe, "_post", lambda *a, **k: fake)
    probe._stream({"model": "m"}, 5.0)
    assert fake.read_calls == 0


def test_stream_timing_spans_first_to_last_content(monkeypatch):
    """Decode must end at the last content token, not at EOF: the usage chunk and
    the connection close arrive afterwards and would be charged to decode."""
    fake = _FakeStream(_sse("a", "b", usage={"completion_tokens": 3}), delay=0.05)
    monkeypatch.setattr(probe, "_post", lambda *a, **k: fake)
    raw = probe._stream({"model": "m"}, 5.0)
    assert raw["first_s"] is not None and raw["last_s"] is not None
    content_span = raw["last_s"] - raw["first_s"]
    to_eof = raw["ended_s"] - raw["first_s"]
    assert content_span > 0.02
    assert to_eof > content_span  # the usage line and EOF come after the last token
    assert raw["usage"]["completion_tokens"] == 3


def test_decode_run_uses_the_content_span_for_the_rate(monkeypatch):
    fake = _FakeStream(
        _sse("a", "b", "c", usage={"completion_tokens": 3, "prompt_tokens": 11}),
        delay=0.05,
    )
    monkeypatch.setattr(probe, "_post", lambda *a, **k: fake)
    monkeypatch.setattr(probe, "spec_snapshot", lambda: {})
    run = probe.decode_run("p", 3, 5.0)
    assert run["completion_tokens"] == 3
    assert run["ttft_s"] > 0.02
    # (3 - 1) tokens over a ~0.1 s content span, not over the longer EOF span.
    assert 10 < run["tok_s"] < 40
    assert run["decode_s"] < run["wall_s"] - run["ttft_s"]


# --- auditor ----------------------------------------------------------------

def _probe_receipt(kind, values, *, valid=None, any_nan=False, any_cache_hit=False):
    key = "tok_s" if kind in ("structured", "essay", "hashmap") else "prefill_tok_s"
    runs = []
    for index, value in enumerate(values, start=1):
        if kind in ("structured", "essay", "hashmap"):
            run = _decode_run(tok_s=value)
        else:
            target = probe.PREFILL_KINDS[kind]
            run = _prefill_run(rate=value, prompt_tokens=target, target=target)
        run["i"] = index
        runs.append(run)
    return {
        "schema": 1,
        "kind": kind,
        "metric": key,
        "runs": runs,
        "invalid_runs": [],
        "valid_runs": len(values) if valid is None else valid,
        f"{key}_median": sorted(values)[len(values) // 2],
        "any_nan": any_nan,
        "any_cache_hit": any_cache_hit,
    }


def _window_receipt(tmp_path, arm_values, *, gates_ok=True, write_probes=True):
    """arm_values: {arm: {lane: [values]}} — one probe file per (arm, lane)."""
    probes = {}
    for arm, lanes in arm_values.items():
        probes[arm] = {}
        for lane, values in lanes.items():
            name = f"task35b-{arm}-{lane}.json"
            if write_probes:
                (tmp_path / name).write_text(json.dumps(_probe_receipt(lane, values)))
            probes[arm][lane] = name
    arms = {}
    for arm in audit.ARMS:
        arms[arm] = {
            "image_tag": audit.ARM_IMAGE[arm],
            "exllamav3_version": audit.ARM_EXLLAMAV3[arm],
            "worker_image_tag": audit.ARM_IMAGE[arm],
            "worker_exllamav3_version": audit.ARM_EXLLAMAV3[arm],
            "jit_stamp": "stamp-b" if arm in ("b", "b2") else "stamp-a",
            "pool_line": "GPU KV cache size: 1,396,551 tokens",
            "pool_capacity": "1396551 tokens; concurrency 1.40x",
            "preemptions_delta": 0,
        }
    gates = {
        "acceptance_rc": 0,
        "memfree_head_gib": 4.1,
        "memfree_worker_gib": 3.3,
        "pool_line": "GPU KV cache size: 1,396,551 tokens",
        "pool_capacity": "1396551 tokens; concurrency 1.40x",
        "pool_capacity_before": "1396551 tokens; concurrency 1.40x",
        "jit_stamp": "stamp-original",
        "jit_stamp_before": "stamp-original",
        "jit_stamp_arm_b": "stamp-b",
    }
    if not gates_ok:
        gates["acceptance_rc"] = 1
    # The runner registers every measurement block before it runs; the judge
    # requires that registry. A healthy window has one successful block per
    # (arm, lane), pointing at the selected probe receipt.
    blocks = [
        {"arm": arm, "lane": lane, "runs": audit.REQUIRED_RUNS[lane], "attempt": "t",
         "path": probes[arm][lane], "ok": True, "returncode": 0, "error": None}
        for arm in audit.ARMS
        for lane in audit.LANES
    ]
    return {
        "schema": 1, "window": "task35b", "arms": arms, "gates": gates, "probes": probes,
        "probe_blocks": blocks,
        "pool_capacity_before": "1396551 tokens; concurrency 1.40x",
    }


def _uniform(value_map):
    """Build arm_values from {arm: {lane: value}}."""
    return {arm: {lane: [value] * audit.REQUIRED_RUNS[lane] for lane, value in lanes.items()}
            for arm, lanes in value_map.items()}


LANE_BASELINE = {"structured": 70.0, "essay": 25.0, "hashmap": 30.0,
                 "prefill60k": 1450.0, "prefill240k": 1400.0}


def _values_with(overrides):
    """Per-arm values: control arms at baseline, candidate arms at overrides."""
    out = {}
    for arm in audit.ARMS:
        base = dict(LANE_BASELINE)
        if arm in audit.CANDIDATE_ARMS:
            base.update(overrides)
        out[arm] = {lane: [value] * audit.REQUIRED_RUNS[lane] for lane, value in base.items()}
    return out


def test_auditor_adopts_a_non_inferior_candidate(tmp_path):
    receipt = _window_receipt(tmp_path, _values_with({"structured": 69.5}))
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ADOPT", result["errors"]
    assert result["lanes"]["structured"]["candidate_over_control"] == round(69.5 / 70.0, 4)


def test_auditor_reverts_a_regressed_lane(tmp_path):
    receipt = _window_receipt(tmp_path, _values_with({"prefill240k": 1200.0}))
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "REVERT"
    assert result["lanes"]["prefill240k"]["verdict"] == "FAIL"
    assert result["lanes"]["structured"]["verdict"] == "PASS"


def test_auditor_reports_a_win_without_requiring_one(tmp_path):
    receipt = _window_receipt(tmp_path, _values_with({"structured": 75.0, "hashmap": 33.0}))
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ADOPT"
    assert result["lanes"]["structured"]["candidate_over_control"] > 1.0


def test_auditor_is_inconclusive_when_the_control_arms_drift(tmp_path):
    values = _values_with({})
    # A2 is the same image as A; a 9% gap means the window moved under itself.
    values["a2"]["essay"] = [27.3] * audit.REQUIRED_RUNS["essay"]
    receipt = _window_receipt(tmp_path, values)
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["lanes"]["essay"]["verdict"] == "INCONCLUSIVE"


def test_auditor_aborts_on_too_few_valid_runs(tmp_path):
    values = _values_with({})
    values["b"]["structured"] = [70.0] * 3
    receipt = _window_receipt(tmp_path, values)
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("§6 requires 9" in message for message in result["errors"])


def test_auditor_aborts_on_a_nan_run(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    (tmp_path / "task35b-b-essay.json").write_text(
        json.dumps(_probe_receipt("essay", [25.0] * 9, any_nan=True))
    )
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("NaN" in message for message in result["errors"])


def test_auditor_aborts_on_a_warm_cold_prefill(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    (tmp_path / "task35b-a-prefill60k.json").write_text(
        json.dumps(_probe_receipt("prefill60k", [1450.0] * 5, any_cache_hit=True))
    )
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("prefix cache" in message for message in result["errors"])


def test_auditor_aborts_when_an_arm_is_not_the_image_it_claims(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    receipt["arms"]["b"]["image_tag"] = audit.ARM_IMAGE["a"]
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("booted" in message for message in result["errors"])


def test_auditor_aborts_when_the_worker_ran_the_wrong_revision(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    receipt["arms"]["b2"]["worker_exllamav3_version"] = "1.4.7"
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("worker reports exllamav3" in message for message in result["errors"])


def test_auditor_aborts_on_a_missing_probe_receipt(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    (tmp_path / "task35b-a2-hashmap.json").unlink()
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("arm a2 lane hashmap" in message for message in result["errors"])


def test_auditor_aborts_on_a_failed_acceptance_battery(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values, gates_ok=False)
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("acceptance battery" in message for message in result["errors"])


def test_window_pool_line_prefers_the_canonical_capacity_line(monkeypatch):
    """A bare `kv_cache` match would hit the startup patch message instead, whose
    value is constant across arms and would make the pool gate vacuous."""
    seen: list[str] = []

    class Fake:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(argv, **_kw):
        command = argv[-1]
        seen.append(command)
        if "GPU KV cache size:" in command:
            return Fake("(EngineCore pid=237) INFO [kv_cache_utils.py:2598] "
                        "GPU KV cache size: 1,396,551 tokens, "
                        "Maximum concurrency for 1,000,000 tokens per request: 1.40x\n")
        return Fake("")

    monkeypatch.setattr(window.win, "run", fake_run)
    raw = window.pool_line_raw()
    assert "GPU KV cache size:" in raw
    assert "GPU KV cache size:" in seen[0]
    # The parsed capacity is what the gate compares, and it carries no PID or
    # timestamp.
    assert window.pool_capacity() == "1396551 tokens; concurrency 1.40x"


def test_pool_capacity_ignores_boot_specific_prefixes(monkeypatch):
    """Two boots reserve an identical pool but log it with different PIDs and
    timestamps. Comparing raw lines would abort a healthy window; comparing the
    parsed capacity must not."""
    boot_a = ("(EngineCore pid=237) INFO 09-11 09:21:22 [kv_cache_utils.py:2598] "
              "GPU KV cache size: 1,396,551 tokens, Maximum concurrency for "
              "1,000,000 tokens per request: 1.40x")
    boot_b = ("(EngineCore pid=91) INFO 09-12 14:02:07 [kv_cache_utils.py:2598] "
              "GPU KV cache size: 1,396,551 tokens, Maximum concurrency for "
              "1,000,000 tokens per request: 1.40x")
    assert boot_a != boot_b
    monkeypatch.setattr(window, "pool_line_raw", lambda: boot_a)
    first = window.pool_capacity()
    monkeypatch.setattr(window, "pool_line_raw", lambda: boot_b)
    assert window.pool_capacity() == first


def test_pool_capacity_distinguishes_a_real_change(monkeypatch):
    monkeypatch.setattr(
        window, "pool_line_raw",
        lambda: "GPU KV cache size: 1,200,000 tokens, "
                "Maximum concurrency for 1,000,000 tokens per request: 1.20x",
    )
    assert window.pool_capacity() == "1200000 tokens; concurrency 1.20x"


def test_pool_capacity_is_empty_when_unparseable(monkeypatch):
    monkeypatch.setattr(window, "pool_line_raw", lambda: "[glm53-kv-capacity-log] 566 ids")
    assert window.pool_capacity() == ""
    monkeypatch.setattr(window, "pool_line_raw", lambda: "")
    assert window.pool_capacity() == ""


def test_window_pool_line_falls_back_to_the_local_capacity_marker(monkeypatch):
    class Fake:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(argv, **_kw):
        if "glm53-kv-capacity-log" in argv[-1]:
            return Fake("[glm53-kv-capacity-log] usable block ids: 566\n")
        return Fake("")

    monkeypatch.setattr(window.win, "run", fake_run)
    assert "glm53-kv-capacity-log" in window.pool_line_raw()


def test_window_pool_line_is_empty_when_neither_line_exists(monkeypatch):
    class Fake:
        stdout = ""

    monkeypatch.setattr(window.win, "run", lambda *_a, **_k: Fake())
    assert window.pool_line_raw() == ""
    assert window.pool_capacity() == ""


# --- review regressions -----------------------------------------------------

def test_auditor_aborts_when_the_kv_pool_capacity_moved(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    receipt["gates"]["pool_capacity"] = "1200000 tokens; concurrency 1.20x"
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("KV pool capacity changed" in message for message in result["errors"])


def test_auditor_aborts_when_one_arm_reserved_a_different_pool(tmp_path):
    """The per-arm check localizes a divergence to the boot that caused it."""
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    receipt["arms"]["b2"]["pool_capacity"] = "1200000 tokens; concurrency 1.20x"
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("arm b2 KV pool capacity differs" in message for message in result["errors"])


def test_auditor_aborts_when_an_arm_recorded_no_pool_capacity(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    del receipt["arms"]["a"]["pool_capacity"]
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("arm a recorded no KV pool capacity" in message for message in result["errors"])


def test_auditor_aborts_when_an_arm_saw_a_preemption(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    receipt["arms"]["b"]["preemptions_delta"] = 1
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("arm b saw 1 preemptions" in message for message in result["errors"])


def test_auditor_aborts_when_the_preemption_delta_is_missing(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    del receipt["arms"]["a2"]["preemptions_delta"]
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("arm a2 recorded no preemption delta" in message for message in result["errors"])


def test_auditor_rejects_a_nan_memory_reading(tmp_path):
    """`NaN < 2.5` is False, so a plain comparison would pass a NaN reading."""
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    receipt["gates"]["memfree_worker_gib"] = float("nan")
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("not a finite memory reading" in message for message in result["errors"])


def test_auditor_rejects_empty_jit_stamps(tmp_path):
    """Two empty stamps must not compare equal and pass."""
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    receipt["gates"]["jit_stamp"] = ""
    receipt["gates"]["jit_stamp_before"] = ""
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("JIT shape stamp was not recorded" in message for message in result["errors"])


def test_auditor_rejects_a_corrupted_excluded_run(tmp_path):
    """A NaN-corrupted observation must not be laundered through the exclusion
    list and then ignored."""
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    path = tmp_path / receipt["probes"]["a"]["structured"]
    payload = json.loads(path.read_text())
    payload["invalid_runs"] = [{"i": 1, "invalid_reason": "NaN/locklock marker in output"}]
    path.write_text(json.dumps(payload))
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("excluded run was corrupted" in message for message in result["errors"])


def test_auditor_aborts_when_the_pre_window_jit_stamp_is_missing(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    receipt["gates"]["jit_stamp_before"] = ""
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("pre-window JIT shape stamp was not recorded" in message
               for message in result["errors"])


def test_auditor_accepts_a_restored_stamp_equal_to_the_pre_window_stamp(tmp_path):
    """Arm B's stamp legitimately differs from the restored one (prod-start.sh
    hashes every raw IMAGE= line), so the restored stamp must be checked against
    the pre-window stamp instead."""
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    assert receipt["gates"]["jit_stamp"] == receipt["gates"]["jit_stamp_before"]
    assert receipt["gates"]["jit_stamp"] != receipt["gates"]["jit_stamp_arm_b"]
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ADOPT"


def test_auditor_is_inconclusive_when_the_candidate_is_unstable(tmp_path):
    """A candidate whose two arms disagree wildly has no settled number to
    compare, even when the control is rock steady."""
    values = _values_with({})
    values["b"]["structured"] = [10.0] * 9
    values["b2"]["structured"] = [40.0] * 9
    receipt = _window_receipt(tmp_path, values)
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "INCONCLUSIVE"
    row = result["lanes"]["structured"]
    assert row["candidate_drift"] > audit.DRIFT_MAX
    assert row["verdict"] == "INCONCLUSIVE"


def test_audit_scope_states_what_adopt_does_not_cover(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ADOPT"
    scope = result["scope"]
    assert any("temp-1" in item for item in scope["does_not_cover"])
    assert "NOT that the full docs/13" in scope["adopt_means"]


def test_auditor_rejects_an_excluded_run_that_is_nan_but_errored(tmp_path):
    """`decode_run_invalid` reports a stream error before the NaN check, so a
    NaN run that also errored carries an error string as its reason. The raw
    `nan` flag must be inspected independently of the exclusion wording."""
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    path = tmp_path / receipt["probes"]["a"]["structured"]
    payload = json.loads(path.read_text())
    payload["invalid_runs"] = [
        {"i": 1, "nan": True, "invalid_reason": "TimeoutError: stream disconnected"}
    ]
    path.write_text(json.dumps(payload))
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("excluded run was corrupted" in message for message in result["errors"])


def test_auditor_rejects_a_missing_per_arm_jit_stamp(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    receipt["arms"]["b2"]["jit_stamp"] = ""
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("arm b2 recorded no JIT shape stamp" in message for message in result["errors"])


def test_auditor_cannot_decide_on_a_bimodal_arm_whose_median_matches(tmp_path):
    """Two candidate arms can agree on their medians while neither is settled.
    A matching median is not evidence of stability. This is a window that
    cannot decide, not an invalid window, so it must not ABORT."""
    values = _values_with({})
    bimodal = [1.0, 1.0, 1.0, 1.0, 25.0, 1000.0, 1000.0, 1000.0, 1000.0]
    values["b"]["structured"] = list(bimodal)
    values["b2"]["structured"] = list(bimodal)
    receipt = _window_receipt(tmp_path, values)
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "INCONCLUSIVE", result["errors"]
    lane = result["lanes"]["structured"]
    assert lane["verdict"] == "INCONCLUSIVE"
    assert lane["unsettled_arms"] == ["b", "b2"]
    assert lane["variability_limit"] == audit.VARIABILITY_MAX
    # Other lanes are still reported, so a partial window stays readable.
    assert result["lanes"]["essay"]["verdict"] == "PASS"


def test_auditor_accepts_a_settled_arm(tmp_path):
    values = _values_with({})
    values["b"]["essay"] = [25.0, 25.2, 24.8, 25.1, 24.9, 25.0, 25.3, 24.7, 25.0]
    receipt = _window_receipt(tmp_path, values)
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ADOPT", result["errors"]
    assert result["lanes"]["essay"]["arm_spread"]["b"] < audit.VARIABILITY_MAX


def test_auditor_cannot_decide_when_a_control_arm_is_unsettled(tmp_path):
    """The variability gate applies to the control arms too: an unsettled
    control cannot anchor the comparison."""
    values = _values_with({})
    values["a2"]["hashmap"] = [10.0, 10.0, 10.0, 10.0, 30.0, 900.0, 900.0, 900.0, 900.0]
    receipt = _window_receipt(tmp_path, values)
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "INCONCLUSIVE", result["errors"]
    assert result["lanes"]["hashmap"]["unsettled_arms"] == ["a2"]


def test_auditor_reports_the_remaining_lanes_of_a_partially_unsettled_window(tmp_path):
    """One unsettled lane must not hide the verdicts of the settled ones."""
    values = _values_with({})
    values["b"]["prefill240k"] = [100.0, 100.0, 100.0, 100.0, 1400.0]
    values["b2"]["prefill240k"] = [100.0, 100.0, 100.0, 100.0, 1400.0]
    receipt = _window_receipt(tmp_path, values)
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "INCONCLUSIVE", result["errors"]
    assert result["lanes"]["prefill240k"]["unsettled_arms"] == ["b", "b2"]
    assert result["lanes"]["structured"]["verdict"] == "PASS"
    assert result["lanes"]["prefill60k"]["verdict"] == "PASS"


def test_auditor_still_aborts_on_a_hard_defect_alongside_an_unsettled_arm(tmp_path):
    """A genuine invalidity outranks noise: corruption is still an ABORT."""
    values = _values_with({})
    values["b"]["structured"] = [1.0, 1.0, 1.0, 1.0, 25.0, 1000.0, 1000.0, 1000.0, 1000.0]
    receipt = _window_receipt(tmp_path, values)
    receipt["arms"]["b2"]["jit_stamp"] = ""
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("recorded no JIT shape stamp" in message for message in result["errors"])


# --- runner -> probe integration --------------------------------------------

def _arm_fixture(monkeypatch, tmp_path, *, warmup_rc=0):
    """Wire phase_arm's collaborators and capture the argv it invokes."""
    monkeypatch.setattr(window, "_RECEIPT", tmp_path / "task35b-window.json")
    monkeypatch.setattr(window, "set_image", lambda tag: None)
    monkeypatch.setattr(window.win, "guarded_start", lambda: None)
    monkeypatch.setattr(window.win, "wait_health", lambda timeout=0: True)
    monkeypatch.setattr(window.win, "memfree_gib", lambda host=None: 4.0)
    monkeypatch.setattr(window, "verify_arm", lambda arm, container, host=None: {
        "image_tag": window.ARMS[arm]["tag"], "exllamav3_version": window.ARMS[arm]["exllamav3"],
    })
    monkeypatch.setattr(window, "jit_stamp", lambda: "stamp")
    monkeypatch.setattr(window, "pool_line_raw", lambda: "GPU KV cache size: 1,396,551 tokens")
    monkeypatch.setattr(window, "pool_capacity", lambda: "1396551 tokens; concurrency 1.40x")
    monkeypatch.setattr(window, "container_started_at", lambda *a, **k: "boot")
    monkeypatch.setattr(window, "preemptions", lambda: 0.0)
    monkeypatch.setattr(window, "save", lambda _s: None)
    seen: list[list[str]] = []

    class Done:
        returncode = warmup_rc
        stdout = ""
        stderr = ""

    monkeypatch.setattr(window.win, "run",
                        lambda argv, **k: seen.append(argv) or Done())
    return seen


def test_arm_warmup_does_not_write_to_a_device_path(monkeypatch, tmp_path):
    """The probe writes atomically (temp + rename), so `--out /dev/null` would
    try to create `/dev/null.tmp` in a root-owned directory and fail the arm."""
    seen = _arm_fixture(monkeypatch, tmp_path)
    window.phase_arm({}, "a")
    warm = [argv for argv in seen if "--kind" in argv]
    assert warm, "no warmup invocation was captured"
    out = warm[0][warm[0].index("--out") + 1]
    assert out != "/dev/null"
    assert not out.startswith("/dev/")
    assert Path(out).parent == tmp_path


def test_arm_records_the_pool_capacity_the_auditor_requires(monkeypatch, tmp_path):
    """The auditor requires `pool_capacity` per arm; the arm phase must produce
    exactly that field, not only the raw line."""
    _arm_fixture(monkeypatch, tmp_path)
    state: dict = {}
    window.phase_arm(state, "a")
    record = state["arms"]["a"]
    assert record["pool_capacity"] == "1396551 tokens; concurrency 1.40x"
    assert record["container_started_at"] == "boot"
    assert record["worker_container_started_at"] == "boot"
    assert record["preemptions_before"] == 0.0


def test_arm_records_feed_the_auditor_without_an_identity_abort(monkeypatch, tmp_path):
    """End-to-end: records actually produced by the arm phase must satisfy the
    auditor's identity and pool checks, so a healthy window is not aborted."""
    _arm_fixture(monkeypatch, tmp_path)
    state: dict = {"pool_capacity_before": "1396551 tokens; concurrency 1.40x"}
    for arm in window.ARMS:
        window.phase_arm(state, arm)
        state["arms"][arm]["preemptions_delta"] = 0
    receipt = {"arms": state["arms"], "pool_capacity_before": state["pool_capacity_before"]}
    errors: list[str] = []
    for arm in audit.ARMS:
        record = receipt["arms"][arm]
        assert record.get("pool_capacity") == receipt["pool_capacity_before"]
        assert record.get("jit_stamp")
        assert record.get("preemptions_delta") == 0
        assert record.get("image_tag") == audit.ARM_IMAGE[arm]
        assert record.get("exllamav3_version") == audit.ARM_EXLLAMAV3[arm]
    assert errors == []


def test_auditor_aborts_when_the_final_jit_stamp_is_not_the_candidate_stamp(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    receipt["gates"]["jit_stamp"] = "stamp-a"
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("JIT shape stamp" in message for message in result["errors"])


# --- window runner wiring ---------------------------------------------------

def test_window_is_a_pre_registered_a_b_b_a_over_two_images():
    assert window.PHASES == (
        "preflight", "disarm",
        "arm_a", "measure_a",
        "arm_b", "measure_b",
        "arm_b2", "measure_b2",
        "arm_a2", "measure_a2",
        "restore", "gates", "rearm", "judge",
    )
    assert [window.ARMS[a]["tag"] for a in ("a", "b", "b2", "a2")] == [
        "glm53-selfbuild:e3-w3-zfill",
        "glm53-selfbuild:e3-w3-zfill-v149",
        "glm53-selfbuild:e3-w3-zfill-v149",
        "glm53-selfbuild:e3-w3-zfill",
    ]
    assert window.PRODUCTION_IMAGE == window.ARMS["b"]["tag"]


def test_window_lane_counts_match_the_auditor_contract():
    assert {lane: runs for lane, (runs, _timeout) in window.LANES.items()} == audit.REQUIRED_RUNS
    assert set(window.LANES) == set(audit.LANES)


def test_window_arm_versions_match_the_auditor_identity():
    for arm, spec in window.ARMS.items():
        assert spec["exllamav3"] == audit.ARM_EXLLAMAV3[arm]
        assert spec["tag"] == audit.ARM_IMAGE[arm]


def test_set_image_appends_a_last_wins_line(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("IMAGE=glm53-selfbuild:e3-w3-zfill-v149\nGLM53_ADAPTIVE_K=ema\n")
    monkeypatch.setattr(window, "ENV_FILE", env)
    window.set_image("glm53-selfbuild:e3-w3-zfill")
    text = env.read_text()
    assert text.count("IMAGE=") == 2
    assert window.effective_env_all()["IMAGE"] == "glm53-selfbuild:e3-w3-zfill"
    assert window.effective_env_all()["GLM53_ADAPTIVE_K"] == "ema"


def test_effective_env_all_ignores_comments_and_bad_keys(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# IMAGE=not-this\n"
        "IMAGE=first\n"
        "IMAGE=second\n"
        "NOT A KEY=value\n"
        'QUOTED="spaced value"\n'
    )
    monkeypatch.setattr(window, "ENV_FILE", env)
    parsed = window.effective_env_all()
    assert parsed["IMAGE"] == "second"
    assert parsed["QUOTED"] == "spaced value"
    assert "NOT A KEY" not in parsed


def test_needs_env_restore_tracks_the_touched_flag_and_the_live_image(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"IMAGE={window.PRODUCTION_IMAGE}\n")
    monkeypatch.setattr(window, "ENV_FILE", env)
    assert window.needs_env_restore({}) is False
    assert window.needs_env_restore({"backup": "x"}) is False
    assert window.needs_env_restore({"backup": "x", "env_touched": True}) is True
    env.write_text("IMAGE=glm53-selfbuild:e3-w3-zfill\n")
    assert window.needs_env_restore({"backup": "x"}) is True


def test_restore_keeps_recovery_intent_until_production_is_verified(monkeypatch, tmp_path):
    """If `guarded_start` fails after the .env is copied back, recovery must
    still believe production needs restoring — otherwise the window leaves
    production down and reports "production was never moved"."""
    env = tmp_path / ".env"
    backup = tmp_path / "backup.env"
    env.write_text("IMAGE=armed\n")
    backup.write_text(f"IMAGE={window.PRODUCTION_IMAGE}\n")
    monkeypatch.setattr(window, "ENV_FILE", env)
    monkeypatch.setattr(window, "save", lambda _s: None)
    monkeypatch.setattr(window.win, "guarded_start", lambda: None)
    monkeypatch.setattr(window.win, "wait_health", lambda timeout=0: False)
    state = {"backup": str(backup), "env_touched": True,
             "env_sha256": window.win.sha256(backup)}
    with pytest.raises(RuntimeError, match="did not become healthy"):
        window.phase_restore(state)
    # The .env is back, but production never came up, so recovery must still act.
    assert state.get("env_touched") is not False
    assert window.needs_env_restore(state) is True


def test_restore_clears_recovery_intent_once_both_nodes_are_verified(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    backup = tmp_path / "backup.env"
    env.write_text("IMAGE=armed\n")
    backup.write_text(f"IMAGE={window.PRODUCTION_IMAGE}\n")
    monkeypatch.setattr(window, "ENV_FILE", env)
    monkeypatch.setattr(window, "save", lambda _s: None)
    monkeypatch.setattr(window.win, "guarded_start", lambda: None)
    monkeypatch.setattr(window.win, "wait_health", lambda timeout=0: True)
    monkeypatch.setattr(window, "verify_arm", lambda arm, container, host=None: {
        "image_tag": window.PRODUCTION_IMAGE, "exllamav3_version": "1.4.9",
    })
    state = {"backup": str(backup), "env_touched": True,
             "env_sha256": window.win.sha256(backup)}
    window.phase_restore(state)
    assert state["env_touched"] is False
    assert state["restored_image_tag"] == window.PRODUCTION_IMAGE


def test_restore_rejects_a_backup_that_does_not_round_trip(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    backup = tmp_path / "backup.env"
    env.write_text("IMAGE=armed\n")
    backup.write_text(f"IMAGE={window.PRODUCTION_IMAGE}\n")
    monkeypatch.setattr(window, "ENV_FILE", env)
    monkeypatch.setattr(window, "save", lambda _s: None)
    state = {"backup": str(backup), "env_touched": True, "env_sha256": "not-the-hash"}
    with pytest.raises(RuntimeError, match="does not match the pre-window hash"):
        window.phase_restore(state)
    assert state["env_touched"] is True


def test_recovery_ok_reports_a_failed_restore():
    assert window._recovery_ok({}) is True
    assert window._recovery_ok({"auto_restore": "ok"}) is True
    assert window._recovery_ok({"auto_restore": "FAILED: boom"}) is False
    assert window._recovery_ok({"timer_restore": "FAILED: boom"}) is False


def test_emergency_restore_stays_armed_after_a_failed_restore(monkeypatch, tmp_path):
    """A failed automatic restore must leave the atexit safety net armed."""
    monkeypatch.setattr(window, "_RESTORE_DONE", False)
    monkeypatch.setattr(window, "_KEEP_ARMED", False)
    monkeypatch.setattr(window, "_ACTIVE", {"backup": "x", "env_touched": True})
    calls: list[str] = []
    monkeypatch.setattr(window, "restore_production", lambda s: calls.append("env"))
    monkeypatch.setattr(window, "recover_timers", lambda s: calls.append("timers"))
    monkeypatch.setattr(window, "save", lambda _s: None)
    window.emergency_restore()
    assert calls == ["env", "timers"]


def test_main_recovers_when_a_save_between_phases_raises(monkeypatch, tmp_path):
    """A SIGTERM landing between phases (or a save() failure) must still trigger
    recovery. Previously the finally-block saw failure=None, skipped the
    restore, and disabled the atexit handler as well."""
    receipt = tmp_path / "w.json"
    saved: list[dict] = []
    real_save = window.save

    def flaky_save(state):
        saved.append(dict(state))
        # Explode on the phase-boundary save of the second phase.
        if len(saved) == 3:
            raise OSError("disk full")
        real_save(state)

    restored: list[str] = []
    monkeypatch.setattr(window, "_RECEIPT", receipt)
    monkeypatch.setattr(window, "save", flaky_save)
    monkeypatch.setattr(window, "restore_production", lambda s: restored.append("env"))
    monkeypatch.setattr(window, "recover_timers", lambda s: restored.append("timers"))
    monkeypatch.setattr(window, "HANDLERS", {
        "preflight": lambda s: None,
        "disarm": lambda s: None,
    })
    monkeypatch.setattr(window, "PHASES", ("preflight", "disarm"))
    monkeypatch.setattr(window, "_RESTORE_DONE", True)
    rc = window.main(["--state", str(receipt)])
    assert rc == 1
    assert restored == ["env", "timers"]


def test_main_recovers_on_a_signal_between_phases(monkeypatch, tmp_path):
    receipt = tmp_path / "w.json"
    monkeypatch.setattr(window, "_RECEIPT", receipt)
    monkeypatch.setattr(window, "save", lambda _s: None)
    restored: list[str] = []
    monkeypatch.setattr(window, "restore_production", lambda s: restored.append("env"))
    monkeypatch.setattr(window, "recover_timers", lambda s: restored.append("timers"))

    def interrupted(_state):
        raise KeyboardInterrupt("signal 15")

    monkeypatch.setattr(window, "HANDLERS", {"preflight": interrupted})
    monkeypatch.setattr(window, "PHASES", ("preflight",))
    assert window.main(["--state", str(receipt)]) == 1
    assert restored == ["env", "timers"]


def test_measure_revalidates_the_running_arm(monkeypatch, tmp_path):
    """A resume with --from measure_a after an automatic recovery would otherwise
    measure the restored production image and file it under arm A."""
    monkeypatch.setattr(window, "save", lambda _s: None)
    monkeypatch.setattr(window, "_RECEIPT", tmp_path / "w.json")
    monkeypatch.setattr(window.win, "memfree_gib", lambda host=None: 4.0)
    probed: list[str] = []
    monkeypatch.setattr(window.subprocess, "run",
                        lambda *a, **k: probed.append("probe") or type(
                            "P", (), {"returncode": 0, "stdout": "", "stderr": ""})())

    def wrong_arm(arm, container, host=None):
        raise RuntimeError(f"arm {arm}: {container} runs 'x', expected 'y'")

    monkeypatch.setattr(window, "verify_arm", wrong_arm)
    with pytest.raises(RuntimeError, match="runs 'x', expected 'y'"):
        window.phase_measure({}, "a")
    assert probed == []  # nothing was measured under the wrong label


def test_measure_rejects_a_boot_that_restarted_since_the_arm_phase(monkeypatch, tmp_path):
    monkeypatch.setattr(window, "save", lambda _s: None)
    monkeypatch.setattr(window, "_RECEIPT", tmp_path / "w.json")
    monkeypatch.setattr(window, "verify_arm", lambda arm, container, host=None: {
        "image_tag": window.ARMS[arm]["tag"], "exllamav3_version": window.ARMS[arm]["exllamav3"],
    })
    monkeypatch.setattr(window, "container_started_at", lambda *a, **k: "2026-09-11T10:00:00Z")
    state = {"arms": {"a": {"container_started_at": "2026-09-11T09:00:00Z",
                            "worker_container_started_at": "2026-09-11T09:00:00Z"}}}
    with pytest.raises(RuntimeError, match="restarted since the arm phase"):
        window.phase_measure(state, "a")


def test_measure_refuses_when_the_boot_identity_is_unavailable(monkeypatch, tmp_path):
    """An unreadable boot token must refuse, not silently bypass the binding."""
    monkeypatch.setattr(window, "save", lambda _s: None)
    monkeypatch.setattr(window, "_RECEIPT", tmp_path / "w.json")
    monkeypatch.setattr(window, "verify_arm", lambda arm, container, host=None: {
        "image_tag": window.ARMS[arm]["tag"], "exllamav3_version": window.ARMS[arm]["exllamav3"],
    })
    monkeypatch.setattr(window, "container_started_at", lambda *a, **k: "")
    state = {"arms": {"a": {"container_started_at": "x", "worker_container_started_at": "x"}}}
    with pytest.raises(RuntimeError, match="could not read the container boot identity"):
        window.phase_measure(state, "a")


def test_measure_refuses_when_the_arm_phase_recorded_no_boot(monkeypatch, tmp_path):
    monkeypatch.setattr(window, "save", lambda _s: None)
    monkeypatch.setattr(window, "_RECEIPT", tmp_path / "w.json")
    monkeypatch.setattr(window, "verify_arm", lambda arm, container, host=None: {
        "image_tag": window.ARMS[arm]["tag"], "exllamav3_version": window.ARMS[arm]["exllamav3"],
    })
    monkeypatch.setattr(window, "container_started_at", lambda *a, **k: "boot")
    with pytest.raises(RuntimeError, match="no boot identity was recorded"):
        window.phase_measure({}, "a")


def test_measure_probe_paths_are_attempt_specific(monkeypatch, tmp_path):
    monkeypatch.setattr(window, "_RECEIPT", tmp_path / "task35b-window-20260911.json")
    monkeypatch.setattr(window, "verify_arm", lambda arm, container, host=None: {
        "image_tag": window.ARMS[arm]["tag"], "exllamav3_version": window.ARMS[arm]["exllamav3"],
    })
    monkeypatch.setattr(window, "container_started_at", lambda *a, **k: "boot")
    monkeypatch.setattr(window.win, "memfree_gib", lambda host=None: 4.0)
    monkeypatch.setattr(window, "preemptions", lambda: 0.0)
    monkeypatch.setattr(window, "save", lambda _s: None)
    seen: list[list[str]] = []

    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(window.subprocess, "run",
                        lambda argv, **k: seen.append(argv) or Done())
    state = {"arms": {"a": {"container_started_at": "boot",
                            "worker_container_started_at": "boot",
                            "preemptions_before": 0.0}}}
    window.phase_measure(state, "a")
    outs = [argv[argv.index("--out") + 1] for argv in seen]
    assert len(outs) == len(window.LANES)
    assert len(set(outs)) == len(outs)  # one distinct file per lane
    for out in outs:
        assert "task35b-window-20260911-a-" in Path(out).name
        assert Path(out).name != "task35b-a-structured.json"


def test_measure_attempts_do_not_collide_across_invocations(monkeypatch, tmp_path):
    """Two measure calls in the same second must not reuse the same paths."""
    monkeypatch.setattr(window, "_RECEIPT", tmp_path / "task35b-window.json")
    monkeypatch.setattr(window, "verify_arm", lambda arm, container, host=None: {
        "image_tag": window.ARMS[arm]["tag"], "exllamav3_version": window.ARMS[arm]["exllamav3"],
    })
    monkeypatch.setattr(window, "container_started_at", lambda *a, **k: "boot")
    monkeypatch.setattr(window.win, "memfree_gib", lambda host=None: 4.0)
    monkeypatch.setattr(window, "preemptions", lambda: 0.0)
    monkeypatch.setattr(window, "save", lambda _s: None)
    seen: list[list[str]] = []

    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(window.subprocess, "run",
                        lambda argv, **k: seen.append(argv) or Done())
    state = {"arms": {"a": {"container_started_at": "boot",
                            "worker_container_started_at": "boot",
                            "preemptions_before": 0.0}}}
    window.phase_measure(state, "a")
    first = {argv[argv.index("--out") + 1] for argv in seen}
    seen.clear()
    window.phase_measure(state, "a")
    second = {argv[argv.index("--out") + 1] for argv in seen}
    assert first and second and not (first & second)


def test_measure_registers_the_block_before_running_it(monkeypatch, tmp_path):
    """A failed block must appear in the receipt, so a retry cannot judge the
    window without seeing that the failure happened."""
    monkeypatch.setattr(window, "_RECEIPT", tmp_path / "task35b-window.json")
    monkeypatch.setattr(window, "verify_arm", lambda arm, container, host=None: {
        "image_tag": window.ARMS[arm]["tag"], "exllamav3_version": window.ARMS[arm]["exllamav3"],
    })
    monkeypatch.setattr(window, "container_started_at", lambda *a, **k: "boot")
    monkeypatch.setattr(window.win, "memfree_gib", lambda host=None: 4.0)
    monkeypatch.setattr(window, "save", lambda _s: None)

    class Failed:
        returncode = 3
        stdout = None
        stderr = None

    monkeypatch.setattr(window.subprocess, "run", lambda *a, **k: Failed())
    state = {"arms": {"a": {"container_started_at": "boot",
                            "worker_container_started_at": "boot",
                            "preemptions_before": 0.0}}}
    with pytest.raises(RuntimeError, match="no output"):
        window.phase_measure(state, "a")
    blocks = state["probe_blocks"]
    assert blocks and blocks[0]["ok"] is False
    assert blocks[0]["returncode"] == 3
    assert blocks[0]["error"] == "(no output)"
    assert state["probes"]["a"] == {}  # nothing registered as usable evidence


def test_wait_quiescent_fails_closed_when_the_job_query_fails(monkeypatch):
    monkeypatch.setattr(window, "_active_services",
                        lambda: {u: "inactive" for u in window.TIMER_SERVICES})
    monkeypatch.setattr(window, "_pending_jobs", lambda: None)
    assert window.wait_quiescent(timeout=0.01, poll=0.0) is False


def test_wait_quiescent_fails_closed_when_a_service_state_is_unknown(monkeypatch):
    monkeypatch.setattr(window, "_active_services",
                        lambda: {u: "unknown(rc=1)" for u in window.TIMER_SERVICES})
    monkeypatch.setattr(window, "_pending_jobs", lambda: 0)
    assert window.wait_quiescent(timeout=0.01, poll=0.0) is False


def test_pending_jobs_reports_none_when_systemctl_fails(monkeypatch):
    class Failed:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(window.win, "run", lambda *a, **k: Failed())
    assert window._pending_jobs() is None


def test_save_is_best_effort_so_receipt_io_cannot_block_recovery(monkeypatch, tmp_path):
    """A persistent receipt-write failure must not stop the window from
    restarting production."""
    monkeypatch.setattr(window, "_RECEIPT", tmp_path / "sub" / "missing" / "w.json")
    state: dict = {}
    window.save(state)  # parent directories do not exist
    assert state["save_failures"] == 1


def test_restore_restarts_production_even_when_the_receipt_cannot_be_written(monkeypatch, tmp_path):
    """The checkpoint write before the restart must not be able to prevent it."""
    env = tmp_path / ".env"
    backup = tmp_path / "backup.env"
    env.write_text("IMAGE=armed\n")
    backup.write_text(f"IMAGE={window.PRODUCTION_IMAGE}\n")
    monkeypatch.setattr(window, "ENV_FILE", env)
    monkeypatch.setattr(window, "_RECEIPT", tmp_path / "no" / "such" / "dir" / "w.json")
    started: list[str] = []
    monkeypatch.setattr(window.win, "guarded_start", lambda: started.append("start"))
    monkeypatch.setattr(window.win, "wait_health", lambda timeout=0: True)
    monkeypatch.setattr(window, "verify_arm", lambda arm, container, host=None: {
        "image_tag": window.PRODUCTION_IMAGE, "exllamav3_version": "1.4.9",
    })
    state = {"backup": str(backup), "env_touched": True,
             "env_sha256": window.win.sha256(backup)}
    window.phase_restore(state)
    assert started == ["start"]
    assert state["env_touched"] is False


def test_disarm_waits_for_quiescence(monkeypatch, tmp_path):
    monkeypatch.setattr(window, "save", lambda _s: None)
    monkeypatch.setattr(window.win, "run", lambda *a, **k: type(
        "P", (), {"returncode": 0, "stdout": "inactive\n", "stderr": ""})())
    monkeypatch.setattr(window, "timer_states", lambda: {u: "inactive" for u in window.TIMERS})
    monkeypatch.setattr(window, "_active_services",
                        lambda: {u: "inactive" for u in window.TIMER_SERVICES})
    monkeypatch.setattr(window, "_pending_jobs", lambda: 0)
    state: dict = {}
    window.phase_disarm(state)
    assert state["quiescent_after_disarm"] is True


def test_disarm_refuses_when_the_watchdog_is_still_running(monkeypatch):
    monkeypatch.setattr(window, "save", lambda _s: None)
    monkeypatch.setattr(window.win, "run", lambda *a, **k: type(
        "P", (), {"returncode": 0, "stdout": "inactive\n", "stderr": ""})())
    monkeypatch.setattr(window, "timer_states", lambda: {u: "inactive" for u in window.TIMERS})
    monkeypatch.setattr(window, "_active_services",
                        lambda: {"vllm-glm53exl3-watchdog.service": "active"})
    monkeypatch.setattr(window, "_pending_jobs", lambda: 0)
    monkeypatch.setattr(window, "wait_quiescent", lambda timeout=0: False)
    with pytest.raises(RuntimeError, match="still active after disarm"):
        window.phase_disarm({})


def test_quiescence_needs_no_pending_jobs(monkeypatch):
    monkeypatch.setattr(window, "_active_services",
                        lambda: {u: "inactive" for u in window.TIMER_SERVICES})
    monkeypatch.setattr(window, "_pending_jobs", lambda: 1)
    assert window.wait_quiescent(timeout=0.01, poll=0.0) is False
    monkeypatch.setattr(window, "_pending_jobs", lambda: 0)
    assert window.wait_quiescent(timeout=1.0, poll=0.0) is True


# --- registered attempts (review round 3, finding 1) ------------------------

def test_auditor_aborts_when_a_failed_attempt_is_hidden_by_a_retry(tmp_path):
    """A resumed arm replaces the selected probe path. The judge must still see
    the failed attempt, so a corrupted block cannot vanish behind a clean one."""
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    # A failed attempt that left a NaN-corrupted receipt behind.
    stale = tmp_path / "task35b-a-structured-attempt1.json"
    stale.write_text(json.dumps({"invalid_runs": [{"i": 1, "nan": True}]}))
    receipt["probe_blocks"].append({
        "arm": "a", "lane": "structured", "runs": 9, "attempt": "t0",
        "path": stale.name, "ok": False, "returncode": 1,
        "error": "probe exited 1",
    })
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT", result["errors"]
    assert any("failed with rc=1" in message for message in result["errors"])
    assert any("NaN-corrupted run" in message for message in result["errors"])


def test_auditor_aborts_when_a_block_was_registered_but_never_completed(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    receipt["probe_blocks"].append({
        "arm": "b", "lane": "essay", "runs": 9, "attempt": "t",
        "path": "task35b-b-essay.json", "ok": None, "returncode": None, "error": None,
    })
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("never completed" in message for message in result["errors"])


def test_auditor_aborts_when_the_receipt_registers_no_blocks(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    del receipt["probe_blocks"]
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ABORT"
    assert any("no registered measurement blocks" in message for message in result["errors"])


def test_auditor_accepts_a_window_whose_blocks_all_succeeded(tmp_path):
    values = _values_with({})
    receipt = _window_receipt(tmp_path, values)
    result = audit.judge(receipt, tmp_path)
    assert result["verdict"] == "ADOPT", result["errors"]


# --- preemption telemetry (review round 3, finding 3) ----------------------

def test_preemptions_fails_closed_on_a_failed_request(monkeypatch):
    class Failed:
        returncode = 22
        stdout = ""
        stderr = "curl: (22) 404"

    monkeypatch.setattr(window.win, "run", lambda *a, **k: Failed())
    assert window.preemptions() is None


def test_preemptions_fails_closed_when_the_metric_is_absent(monkeypatch):
    class Ok:
        returncode = 0
        stdout = "vllm:num_requests_running{engine=\"0\"} 0.0\n"

    monkeypatch.setattr(window.win, "run", lambda *a, **k: Ok())
    assert window.preemptions() is None


def test_preemptions_fails_closed_on_a_non_finite_counter(monkeypatch):
    class Ok:
        returncode = 0
        stdout = "vllm:num_preemptions_total{engine=\"0\"} nan\n"

    monkeypatch.setattr(window.win, "run", lambda *a, **k: Ok())
    assert window.preemptions() is None


def test_preemptions_sums_finite_counters(monkeypatch):
    class Ok:
        returncode = 0
        stdout = ('vllm:num_preemptions_total{engine="0"} 2.0\n'
                  'vllm:num_preemptions_total{engine="1"} 3.0\n')

    monkeypatch.setattr(window.win, "run", lambda *a, **k: Ok())
    assert window.preemptions() == 5.0


def test_measure_refuses_when_preemption_telemetry_is_unavailable(monkeypatch, tmp_path):
    """Two unavailable samples must not read as an accepted zero delta."""
    monkeypatch.setattr(window, "_RECEIPT", tmp_path / "task35b-window.json")
    monkeypatch.setattr(window, "verify_arm", lambda arm, container, host=None: {
        "image_tag": window.ARMS[arm]["tag"], "exllamav3_version": window.ARMS[arm]["exllamav3"],
    })
    monkeypatch.setattr(window, "container_started_at", lambda *a, **k: "boot")
    monkeypatch.setattr(window.win, "memfree_gib", lambda host=None: 4.0)
    monkeypatch.setattr(window, "preemptions", lambda: None)
    monkeypatch.setattr(window, "save", lambda _s: None)

    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(window.subprocess, "run", lambda *a, **k: Done())
    state = {"arms": {"a": {"container_started_at": "boot",
                            "worker_container_started_at": "boot",
                            "preemptions_before": 0.0}}}
    with pytest.raises(RuntimeError, match="preemption telemetry unavailable"):
        window.phase_measure(state, "a")
    assert state["arms"]["a"]["preemptions_delta"] is None


# --- quiescence and timer state use the return code (round 3) --------------

def test_active_services_marks_a_missing_unit_unknown(monkeypatch):
    """`is-active` prints `inactive` for a nonexistent unit with rc=4 (verified
    on the live node), so stdout alone cannot distinguish "disarmed" from
    "asked the wrong user manager"."""
    class Missing:
        returncode = 4
        stdout = "inactive"

    monkeypatch.setattr(window.win, "run", lambda *a, **k: Missing())
    states = window._active_services()
    assert all("unknown(rc=4" in value for value in states.values())
    assert window.wait_quiescent(timeout=0.01, poll=0.0) is False


def test_active_services_accepts_a_genuine_inactive_unit(monkeypatch):
    class Inactive:
        returncode = 3
        stdout = "inactive"

    monkeypatch.setattr(window.win, "run", lambda *a, **k: Inactive())
    assert set(window._active_services().values()) == {"inactive"}


def test_timer_states_marks_a_missing_unit_unknown(monkeypatch):
    class Missing:
        returncode = 4
        stdout = "inactive"

    monkeypatch.setattr(window.win, "run", lambda *a, **k: Missing())
    assert all("unknown(rc=4" in value for value in window.timer_states().values())


def test_disarm_refuses_when_a_timer_is_not_provably_stopped(monkeypatch):
    """A missing unit must not read as a successful stop: `stop` can succeed
    (rc=0) while `is-active` reports rc=4 because the unit does not exist."""
    monkeypatch.setattr(window, "save", lambda _s: None)

    class Result:
        def __init__(self, returncode, stdout):
            self.returncode, self.stdout, self.stderr = returncode, stdout, ""

    def fake_run(argv, **kwargs):
        # `stop` succeeds; the follow-up `is-active` cannot find the unit.
        return Result(0, "") if argv[2] == "stop" else Result(4, "inactive")

    monkeypatch.setattr(window.win, "run", fake_run)
    with pytest.raises(RuntimeError, match="not provably stopped"):
        window.phase_disarm({})


def test_rearm_attempts_both_timers_even_when_the_first_fails(monkeypatch):
    """A persistent watchdog failure must not prevent the metrics timer from
    being restored."""
    monkeypatch.setattr(window, "save", lambda _s: None)
    started: list[str] = []

    class Start:
        def __init__(self, unit, status="inactive"):
            self.unit, self.returncode, self.stdout, self.stderr = unit, 0, status, ""

    class Failed:
        returncode = 1
        stdout = "inactive"
        stderr = "boom"

    def fake_run(argv, **kwargs):
        unit = argv[-1]
        if argv[2] == "start":
            started.append(unit)
            return Failed() if unit == window.TIMERS[0] else Start(unit)
        # `is-active`: the first timer never came up, the second did.
        return Start(unit, "inactive" if unit == window.TIMERS[0] else "active")

    monkeypatch.setattr(window.win, "run", fake_run)
    state: dict = {}
    with pytest.raises(RuntimeError, match="timers not restored"):
        window.phase_rearm(state)
    # Both units were attempted, despite the first failing.
    assert started == list(window.TIMERS)
    failure = state["rearm_failures"][window.TIMERS[0]]
    assert failure.startswith("start exited 1")
    assert "not active after start" in failure
    assert window.TIMERS[1] not in state["rearm_failures"]


# --- resume must not skip disarm (review round 3, finding 2) ---------------

def _main_fixture(monkeypatch, tmp_path):
    """A hermetic `main()`: no real phases, no real restore at exit."""
    receipt = tmp_path / "task35b-window.json"
    receipt.write_text(json.dumps({"backup": "b.env"}))
    monkeypatch.setattr(window, "save", lambda _s: None)
    monkeypatch.setattr(window, "emergency_restore", lambda *a, **k: None)
    monkeypatch.setattr(window, "recover_timers", lambda state: None)
    monkeypatch.setattr(window, "HANDLERS",
                        {name: (lambda state: None) for name in window.PHASES})
    return receipt


def test_arm_phases_are_exactly_the_operational_phases():
    assert set(window.ARM_PHASES) == {
        "arm_a", "measure_a", "arm_b", "measure_b",
        "arm_b2", "measure_b2", "arm_a2", "measure_a2",
    }
    assert "disarm" not in window.ARM_PHASES
    assert "restore" not in window.ARM_PHASES


def test_require_disarmed_refuses_when_the_timers_are_still_armed(monkeypatch):
    class Armed:
        returncode = 0
        stdout = "active"
        stderr = ""

    monkeypatch.setattr(window.win, "run", lambda *a, **k: Armed())
    with pytest.raises(RuntimeError, match="not disarmed"):
        window.require_disarmed("--from arm_b")


def test_require_disarmed_refuses_when_a_timer_is_unreadable(monkeypatch):
    class Missing:
        returncode = 4
        stdout = "inactive"
        stderr = ""

    monkeypatch.setattr(window.win, "run", lambda *a, **k: Missing())
    with pytest.raises(RuntimeError, match="not disarmed"):
        window.require_disarmed("--from arm_b")


def test_require_disarmed_refuses_when_the_services_are_not_quiescent(monkeypatch):
    class Inactive:
        returncode = 3
        stdout = "inactive"
        stderr = ""

    monkeypatch.setattr(window.win, "run", lambda *a, **k: Inactive())
    monkeypatch.setattr(window, "_pending_jobs", lambda: 2)
    with pytest.raises(RuntimeError, match="not quiescent"):
        window.require_disarmed("--from measure_b", timeout=0.01, poll=0.0)


def test_require_disarmed_passes_when_timers_are_stopped_and_quiet(monkeypatch):
    class Inactive:
        returncode = 3
        stdout = "inactive"
        stderr = ""

    monkeypatch.setattr(window.win, "run", lambda *a, **k: Inactive())
    monkeypatch.setattr(window, "_pending_jobs", lambda: 0)
    window.require_disarmed("--from arm_b")


def test_main_checks_disarm_when_resuming_an_operational_phase(monkeypatch, tmp_path):
    """`--from arm_b` must re-establish the disarm prerequisite, because
    automatic recovery re-arms the timers."""
    receipt = _main_fixture(monkeypatch, tmp_path)
    seen: list[str] = []

    def refuse(where):
        seen.append(where)
        raise RuntimeError(f"{where}: not disarmed")

    monkeypatch.setattr(window, "require_disarmed", refuse)
    rc = window.main(["--state", str(receipt), "--from", "arm_b", "--to", "arm_b"])
    assert rc == 2
    assert seen == ["--from arm_b"]


def test_main_checks_disarm_for_every_operational_phase(monkeypatch, tmp_path):
    receipt = _main_fixture(monkeypatch, tmp_path)
    seen: list[str] = []
    monkeypatch.setattr(window, "require_disarmed",
                        lambda where: seen.append(where) or None)
    rc = window.main(["--state", str(receipt), "--from", "arm_b", "--to", "arm_b"])
    assert rc == 0
    assert seen == ["--from arm_b"]


def test_main_does_not_check_disarm_for_post_window_phases(monkeypatch, tmp_path):
    """The phases after `restore` legitimately run with the timers re-armed."""
    receipt = _main_fixture(monkeypatch, tmp_path)
    called: list[str] = []
    monkeypatch.setattr(window, "require_disarmed", lambda where: called.append(where))
    rc = window.main(["--state", str(receipt), "--from", "gates", "--to", "gates"])
    assert called == []
    assert rc == 0


def test_main_does_not_check_disarm_when_starting_from_disarm(monkeypatch, tmp_path):
    receipt = _main_fixture(monkeypatch, tmp_path)
    called: list[str] = []
    monkeypatch.setattr(window, "require_disarmed", lambda where: called.append(where))
    rc = window.main(["--state", str(receipt), "--from", "disarm", "--to", "disarm"])
    assert called == []
    assert rc == 0


def test_judge_writes_the_audit_next_to_the_window_receipt(monkeypatch, tmp_path):
    receipt = tmp_path / "task35b-window-20260911-000000.json"
    receipt.write_text(json.dumps({"arms": {}, "gates": {}, "probes": {}}))
    monkeypatch.setattr(window, "_RECEIPT", receipt)
    state: dict = {}
    window.phase_judge(state)
    assert state["verdict"] == "ABORT"
    assert state["audit_receipt"] == "task35b-window-20260911-000000-audit.json"
    assert (tmp_path / state["audit_receipt"]).is_file()
    # `_RECEIPT` is the state checkpoint, so save() rewrites it with window
    # state; the audit must live in its own file, identified by the keys only
    # the auditor writes.
    assert "errors" in json.loads((tmp_path / state["audit_receipt"]).read_text())
    assert "errors" not in json.loads(receipt.read_text())


def test_judge_keeps_the_audit_distinct_for_a_custom_receipt_name(monkeypatch, tmp_path):
    """A receipt without a `-window-` token previously collided: the auditor
    wrote over the window receipt and the next save() overwrote the audit,
    destroying both."""
    receipt = tmp_path / "custom.json"
    receipt.write_text(json.dumps({"arms": {}, "gates": {}, "probes": {}}))
    monkeypatch.setattr(window, "_RECEIPT", receipt)
    state: dict = {"marker": "window-state"}
    window.phase_judge(state)
    assert state["audit_receipt"] == "custom-audit.json"
    assert state["audit_receipt"] != "custom.json"
    audit_payload = json.loads((tmp_path / "custom-audit.json").read_text())
    assert audit_payload["verdict"] == "ABORT"
    assert "errors" in audit_payload
    # The checkpoint still holds window state and did not become the audit.
    window_payload = json.loads(receipt.read_text())
    assert window_payload["marker"] == "window-state"
    assert window_payload["audit_receipt"] == "custom-audit.json"
    assert "errors" not in window_payload


# --- ISA probe (task 38 item 2) ---------------------------------------------

isa = _load(SCRIPTS / "probe_isa_e2m1_silicon.py", "glm53_probe_isa")


def test_isa_reference_follows_the_ptx_nibble_order():
    # PTX ISA: `a` is converted into the upper nibble, `b` into the lower.
    # 1.0 -> code 2, 2.0 -> code 4.
    assert isa.expected_pair(1.0, 2.0) == 0x24
    assert isa.reversed_pair(1.0, 2.0) == 0x42
    # The task 38 record shows 0x42 for this pair, i.e. the original probe fed
    # (2.0, 1.0) -- or its inputs were ordered opposite to the PTX spec.
    assert isa.expected_pair(2.0, 1.0) == 0x42


@pytest.mark.parametrize(
    "value,code",
    [
        (0.0, 0b0000), (0.5, 0b0001), (1.0, 0b0010), (1.5, 0b0011),
        (2.0, 0b0100), (3.0, 0b0101), (4.0, 0b0110), (6.0, 0b0111),
        (-1.0, 0b1010), (-6.0, 0b1111),
    ],
)
def test_isa_e2m1_codes(value, code):
    assert isa.e2m1_code(value) == code


def test_isa_e2m1_rounds_to_nearest_even_on_a_tie():
    assert isa.e2m1_code(0.25) == 0b0000  # tie 0.0/0.5 -> even code 0
    assert isa.e2m1_code(0.75) == 0b0010  # tie 0.5/1.0 -> even code 2
    assert isa.e2m1_code(0.4) == 0b0001
    assert isa.e2m1_code(0.6) == 0b0001
    assert isa.e2m1_code(2.7) == 0b0101


def test_isa_e2m1_saturates_and_rejects_nan():
    assert isa.e2m1_code(1e30) == 0b0111
    assert isa.e2m1_code(-1e30) == 0b1111
    with pytest.raises(ValueError):
        isa.e2m1_code(float("nan"))


def test_isa_ptx_uses_one_parameter_fed_kernel():
    ptx = isa.build_ptx()
    assert ".target sm_121a" in ptx
    assert ".entry probe_e2m1" in ptx
    # One kernel, one conversion: the cases are separate launches.
    assert ptx.count("cvt.rn.satfinite.e2m1x2.f32") == 1
    # Operands must arrive as parameters, never as `mov.f32` immediates: ptxas
    # mis-materializes the immediates and every case then reads 0x00.
    assert "ld.param.f32" in ptx
    assert "mov.f32" not in ptx
    assert "0f" not in ptx
    # The destination is a single byte.
    assert ".reg .b8" in ptx
    assert "st.global.u8" in ptx


def test_isa_judge_confirms_spec_correct_silicon():
    observed = [isa.expected_pair(a, b) for _n, a, b, _t in isa.CASES]
    result = isa.judge(observed)
    assert result["verdict"] == "SILICON_CONFIRMED"
    assert result["errors"] == []
    assert result["reversed_order_cases"] == 0
    assert result["cases_matching_spec"] == len(isa.CASES)


def test_isa_judge_rejects_a_reversed_nibble_order():
    observed = [isa.reversed_pair(a, b) for _n, a, b, _t in isa.CASES]
    result = isa.judge(observed)
    assert result["verdict"] == "SILICON_MISMATCH"
    assert result["reversed_order_cases"] > 0
    assert any("REVERSED nibble order" in message for message in result["errors"])


def test_isa_judge_rejects_a_wrong_conversion():
    observed = [isa.expected_pair(a, b) for _n, a, b, _t in isa.CASES]
    observed[0] = 0x00
    result = isa.judge(observed)
    assert result["verdict"] == "SILICON_MISMATCH"
    assert any("recorded-reversed-2.0-1.0" in message for message in result["errors"])


def test_isa_judge_rejects_an_all_zero_result():
    """The exact failure the immediate-operand form produced must not pass.

    `(0.0, 0.0)` really does encode as 0x00, so that one case matches; the
    discriminating cases must still be reported as mismatches.
    """
    result = isa.judge([0x00] * len(isa.CASES))
    assert result["verdict"] == "SILICON_MISMATCH"
    assert result["cases_matching_spec"] == 1
    flagged = {message.split(" ")[0] for message in result["errors"]}
    assert {"distinct-1.0-2.0", "top-of-range", "negative"} <= flagged
    assert "zero" not in flagged


def test_isa_ties_are_recorded_but_not_gated():
    rows = {row["case"]: row for row in isa.judge([0x00] * len(isa.CASES))["cases"]}
    assert rows["tie-0.25-0.75"]["tie"] is True
    assert isa.judge([0x00] * len(isa.CASES))["gated_cases"] == len(isa.CASES) - 1


def test_isa_judge_rejects_a_short_result():
    with pytest.raises(RuntimeError):
        isa.judge([0x00] * (len(isa.CASES) - 1))


def test_isa_ptx_only_mode_needs_no_gpu(monkeypatch, tmp_path):
    def explode(*_a, **_k):
        raise AssertionError("--ptx-only must not touch the driver")

    monkeypatch.setattr(isa, "run_on_gpu", explode)
    assert isa.main(["--ptx-only", "--workdir", str(tmp_path)]) == 0
    assert (tmp_path / "run_probe.ptx").is_file()


def test_isa_reports_probe_unavailable_when_the_driver_refuses(monkeypatch, tmp_path):
    def refuse(_ptx, _cases=isa.CASES):
        raise RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")

    monkeypatch.setattr(isa, "run_on_gpu", refuse)
    out = tmp_path / "isa.json"
    assert isa.main(["--out", str(out)]) == 2
    payload = json.loads(out.read_text())
    assert payload["verdict"] == "PROBE_UNAVAILABLE"
    assert "OUT_OF_MEMORY" in payload["error"]


def test_isa_main_writes_a_silicon_confirmed_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(isa, "run_on_gpu", lambda ptx, cases=isa.CASES: [
        isa.expected_pair(a, b) for _n, a, b, _t in cases
    ])
    out = tmp_path / "isa.json"
    assert isa.main(["--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["verdict"] == "SILICON_CONFIRMED"
    assert payload["probe"] == "cvt.rn.satfinite.e2m1x2.f32"


def test_window_has_no_isa_phase():
    """The probe runs alongside production (1-byte alloc + 1-thread launch
    succeed with the serving stack up), so the window must not stop production
    for it."""
    assert "isa" not in window.PHASES
    assert not hasattr(window, "ISA_PROBE")
    assert not hasattr(window, "phase_isa")
