"""Guards for the window runner and the serving smoke.

Both scripts decide whether a candidate is adopted, so their negative paths
matter as much as their happy path: a comparison that reports ADOPT over a
failed, unhealthy, incoherent, unarmed or memory-starved probe is worse than one
that reports nothing, and a smoke that accepts a malformed tool call is not
serving-correctness evidence.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WINDOW = ROOT / "scripts" / "run_shared_experts_window.py"
SMOKE = ROOT / "scripts" / "smoke_shared_experts.py"
KNOB = "GLM53_SHARED_EXPERTS_EARLY"


def _load(path: Path, tag: str):
    spec = importlib.util.spec_from_file_location(f"win_{tag}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def window():
    return _load(WINDOW, "runner")


@pytest.fixture(scope="module")
def smoke():
    return _load(SMOKE, "smoke")


# --------------------------------------------------------------------------
# the window runner's gate
# --------------------------------------------------------------------------
LANES = {"structured": 71.15, "prose": 31.84, "essay": 24.09}


def _receipt(*, armed: bool, scale: float = 1.0) -> dict:
    """A receipt that satisfies every prerequisite, before any mutation."""
    lanes = []
    for lane, value in LANES.items():
        lanes.append(
            {
                "lane": lane,
                "returncode": 0,
                "tok_s_median": value * scale,
                "any_nan": False,
                "coherent": None if lane == "structured" else True,
            }
        )
    return {
        "phase": "armed" if armed else "control",
        "health_before": 200,
        "health_after": 200,
        "effective_env": {KNOB: "1"} if armed else {"GLM53_ROUTER_ONCE": "1"},
        "marker_counts": {"head": 3, "worker": 3}
        if armed
        else {"head": 0, "worker": 0},
        "memfree": {
            "head": {"min_kib": 4 * 1024 * 1024, "samples": 40},
            "worker": {"min_kib": 4 * 1024 * 1024, "samples": 40},
        },
        "lanes": lanes,
    }


def _smoke(requests: int = 8) -> dict:
    return {
        "ok": True,
        "failed": 0,
        "any_nan": False,
        "requests": requests,
        "results": [{"name": f"r{i}", "ok": True} for i in range(requests)],
        "log_errors": {"head": [], "worker": []},
        "log_failures": {"head": "", "worker": ""},
    }


def _write(tmp: Path, name: str, payload: object) -> str:
    path = tmp / name
    path.write_text(json.dumps(payload))
    return str(path)


def _verdict(
    window, tmp: Path, control: object, armed: object, smoke: object = None, capsys=None
) -> dict:
    """Drive the compare() entry point and return the verdict it emitted."""
    args = types.SimpleNamespace(
        compare=[_write(tmp, "c.json", control), _write(tmp, "a.json", armed)],
        smoke=_write(tmp, "s.json", smoke) if smoke is not None else None,
        out=None,
    )
    assert window.compare(args) == 0
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def verdict(window, tmp_path, capsys):
    def run(control: object, armed: object, smoke: object = None) -> dict:
        return _verdict(window, tmp_path, control, armed, smoke, capsys)

    return run


def test_clean_probe_with_the_smoke_has_no_gate_failures(window) -> None:
    assert (
        window.gate_failures(_receipt(armed=False), _receipt(armed=True), _smoke())
        == []
    )


def test_a_missing_smoke_is_a_partial_assessment(window, verdict) -> None:
    """The smoke is part of the gate, so its absence must not read as ADOPT."""
    out = verdict(_receipt(armed=False), _receipt(armed=True, scale=1.03))
    assert out["verdict"] == "PARTIAL-GATE"
    assert out["complete"] is False
    assert any("smoke" in problem for problem in out["gate_failures"])


def test_a_failed_smoke_is_invalid(window, verdict) -> None:
    bad = _smoke()
    bad["failed"] = 1
    out = verdict(_receipt(armed=False), _receipt(armed=True, scale=1.03), bad)
    assert out["verdict"] == "INVALID"


def test_a_smoke_with_uncollected_logs_is_invalid(window) -> None:
    bad = _smoke()
    bad["log_failures"]["worker"] = "exit 255: Connection refused"
    assert window.smoke_failures(bad)


def test_a_smoke_with_engine_errors_is_invalid(window) -> None:
    bad = _smoke()
    bad["log_errors"]["head"] = ["CUDA error"]
    assert window.smoke_failures(bad)


def test_a_smoke_without_log_status_is_invalid(window) -> None:
    bad = _smoke()
    del bad["log_failures"]
    assert window.smoke_failures(bad)


def test_a_smoke_that_made_no_requests_is_not_clean(window, verdict) -> None:
    """`--repeat 0` yields a receipt that is ok, empty and worthless."""
    empty = _smoke(requests=0)
    assert empty["requests"] == 0 and empty["results"] == []
    assert window.smoke_failures(empty)
    out = verdict(_receipt(armed=False), _receipt(armed=True, scale=1.03), empty)
    assert out["verdict"] == "INVALID"


def test_a_smoke_without_recorded_results_is_not_clean(window) -> None:
    for mutate in (
        lambda s: s.pop("requests"),
        lambda s: s.pop("results"),
        lambda s: s.update(results=[]),
        lambda s: s.update(results=[{"ok": True}]),
        lambda s: s.update(results=[{"name": "tool-weather", "ok": False}]),
        lambda s: s.update(requests="8"),
    ):
        bad = _smoke()
        mutate(bad)
        assert window.smoke_failures(bad), bad


def test_a_smoke_missing_a_rank_status_is_not_clean(window) -> None:
    for node in ("head", "worker"):
        bad = _smoke()
        del bad["log_errors"][node]
        assert window.smoke_failures(bad), node
        bad = _smoke()
        del bad["log_failures"][node]
        assert window.smoke_failures(bad), node


def test_improvement_is_adoptable(window, verdict) -> None:
    control = _receipt(armed=False)
    armed = _receipt(armed=True, scale=1.03)
    assert window.gate_failures(control, armed, _smoke()) == []
    out = verdict(control, armed, _smoke())
    assert out["verdict"] == "ADOPT"
    assert out["complete"] is True
    assert all(row["parity"] and row["above_floor"] for row in out["rows"])


def _fails(window, control: object, armed: object, smoke: object = None) -> bool:
    return bool(
        window.gate_failures(control, armed, _smoke() if smoke is None else smoke)
    )


def test_unhealthy_pair_is_not_a_result(window) -> None:
    control = _receipt(armed=False)
    armed = _receipt(armed=True)
    armed["health_before"] = 0
    assert _fails(window, control, armed)
    armed = _receipt(armed=True)
    armed["health_after"] = 503
    assert _fails(window, control, armed)


def test_failed_lane_is_not_a_result(window) -> None:
    control = _receipt(armed=False)
    armed = _receipt(armed=True)
    armed["lanes"][1]["returncode"] = 2
    armed["lanes"][1]["tok_s_median"] = None
    assert _fails(window, control, armed)


def test_nan_is_not_a_result(window) -> None:
    control = _receipt(armed=False)
    armed = _receipt(armed=True)
    armed["lanes"][2]["any_nan"] = True
    assert _fails(window, control, armed)


def test_unconfirmed_coherence_is_not_a_result(window) -> None:
    control = _receipt(armed=False)
    armed = _receipt(armed=True)
    armed["lanes"][1]["coherent"] = False
    assert _fails(window, control, armed)
    armed = _receipt(armed=True)
    armed["lanes"][2]["coherent"] = None
    assert _fails(window, control, armed)


def test_missing_memfree_samples_are_not_a_result(window) -> None:
    control = _receipt(armed=False)
    armed = _receipt(armed=True)
    armed["memfree"]["worker"] = {"min_kib": None}
    assert _fails(window, control, armed)
    armed = _receipt(armed=True)
    armed["memfree"] = {}
    assert _fails(window, control, armed)


def test_memfree_below_the_floor_is_not_a_result(window) -> None:
    control = _receipt(armed=False)
    armed = _receipt(armed=True)
    armed["memfree"]["head"]["min_kib"] = 1024
    assert _fails(window, control, armed)


def test_an_unarmed_boot_is_not_a_result(window) -> None:
    """The treatment must be provably active before its numbers count."""
    control = _receipt(armed=False)
    armed = _receipt(armed=True)
    armed["marker_counts"] = {"head": 0, "worker": 0}
    assert _fails(window, control, armed)
    armed = _receipt(armed=True)
    armed["effective_env"] = {"GLM53_ROUTER_ONCE": "1"}
    assert _fails(window, control, armed)
    armed = _receipt(armed=True)
    armed["effective_env"] = {KNOB: "0"}
    assert _fails(window, control, armed)


def test_a_patched_control_is_not_a_result(window) -> None:
    """The control must be the unarmed boot, not a second armed one."""
    control = _receipt(armed=False)
    control["marker_counts"] = {"head": 3, "worker": 3}
    assert _fails(window, control, _receipt(armed=True))


def test_a_regressing_lane_is_not_adoptable(window) -> None:
    control = _receipt(armed=False)
    armed = _receipt(armed=True)
    armed["lanes"][1]["tok_s_median"] = LANES["prose"] * 0.90
    assert window.gate_failures(control, armed, _smoke()) == []
    _rows, verdict = window.compare_rows(control, armed)
    assert verdict == "REVERT"


# --------------------------------------------------------------------------
# malformed receipts must be refused, not silently read as clean
# --------------------------------------------------------------------------
def test_absent_nan_evidence_is_not_clean_evidence(window) -> None:
    control = _receipt(armed=False)
    armed = _receipt(armed=True)
    for row in armed["lanes"]:
        del row["any_nan"]
    assert _fails(window, control, armed)


def test_zero_sample_counts_are_not_evidence(window, verdict) -> None:
    control = _receipt(armed=False)
    armed = _receipt(armed=True)
    armed["memfree"]["head"]["samples"] = 0
    assert _fails(window, control, armed)
    assert verdict(control, armed, _smoke())["verdict"] == "INVALID"
    armed = _receipt(armed=True)
    del armed["memfree"]["worker"]["samples"]
    assert _fails(window, control, armed)
    assert verdict(control, armed, _smoke())["verdict"] == "INVALID"


def test_a_missing_required_field_is_refused(window, verdict) -> None:
    control = _receipt(armed=False)
    for mutate in (
        lambda r: r.pop("health_before"),
        lambda r: r.pop("health_after"),
        lambda r: r.pop("marker_counts"),
        lambda r: r.pop("memfree"),
        lambda r: r.pop("effective_env"),
        lambda r: r.pop("lanes"),
    ):
        armed = _receipt(armed=True)
        mutate(armed)
        assert _fails(window, control, armed), mutate
        assert verdict(control, armed, _smoke())["verdict"] == "INVALID", mutate


def test_malformed_shapes_are_refused_not_crashed(window, verdict) -> None:
    """A lane mapping or a non-list must yield a verdict, not an exception."""
    control = _receipt(armed=False)
    for armed in (
        {"lanes": {"structured": 71.0}},
        {"lanes": "structured"},
        {"lanes": [None]},
        "not a receipt",
        None,
        7,
        [],
    ):
        assert _fails(window, control, armed), armed
        # and the entry point must still emit a verdict rather than raise
        out = verdict(control, armed, _smoke())
        assert out["verdict"] == "INVALID", armed
    # a malformed *control* must not crash the row calculation either
    for control in ("not a receipt", None, {"lanes": "structured"}):
        out = verdict(control, _receipt(armed=True), _smoke())
        assert out["verdict"] == "INVALID", control


def test_a_duplicate_lane_is_refused(window, verdict) -> None:
    """A second row for a lane must not be able to shadow a failing one."""
    control = _receipt(armed=False)
    armed = _receipt(armed=True, scale=1.03)
    duplicate = {
        "lane": "prose",
        "returncode": 2,
        "tok_s_median": LANES["prose"] * 0.5,
        "any_nan": True,
        "coherent": False,
    }
    armed["lanes"].append(duplicate)
    assert _fails(window, control, armed)
    out = verdict(control, armed, _smoke())
    assert out["verdict"] == "INVALID"
    assert any("expected exactly 1" in problem for problem in out["gate_failures"])
    # the same duplicate, otherwise healthy, is still refused
    armed = _receipt(armed=True, scale=1.03)
    armed["lanes"].append(dict(armed["lanes"][1]))
    assert _fails(window, control, armed)


def test_a_non_object_lane_entry_is_refused(window, verdict) -> None:
    control = _receipt(armed=False)
    armed = _receipt(armed=True, scale=1.03)
    armed["lanes"].append(None)
    assert _fails(window, control, armed)
    out = verdict(control, armed, _smoke())
    assert out["verdict"] == "INVALID"
    assert any("is not an object" in problem for problem in out["gate_failures"])


def test_a_lane_entry_without_a_name_is_refused(window) -> None:
    control = _receipt(armed=False)
    armed = _receipt(armed=True)
    armed["lanes"].append({"returncode": 0, "tok_s_median": 99.0})
    assert _fails(window, control, armed)


def test_a_zero_or_invalid_median_is_refused(window, verdict) -> None:
    control = _receipt(armed=False)
    armed = _receipt(armed=True)
    armed["lanes"][1]["tok_s_median"] = 0
    assert _fails(window, control, armed)
    out = verdict(control, armed, _smoke())
    assert out["verdict"] == "INVALID"
    for bad in (None, "31.8", float("nan"), float("inf"), True, -1):
        armed = _receipt(armed=True)
        armed["lanes"][1]["tok_s_median"] = bad
        assert _fails(window, control, armed), bad


def test_a_missing_lane_is_refused(window) -> None:
    control = _receipt(armed=False)
    armed = _receipt(armed=True)
    armed["lanes"] = [row for row in armed["lanes"] if row["lane"] != "essay"]
    assert _fails(window, control, armed)


def test_an_unexpected_lane_is_refused(window) -> None:
    control = _receipt(armed=False)
    armed = _receipt(armed=True)
    armed["lanes"].append(
        {
            "lane": "prefill",
            "returncode": 0,
            "tok_s_median": 999.0,
            "any_nan": False,
            "coherent": True,
        }
    )
    assert _fails(window, control, armed)


def test_a_zero_control_median_does_not_crash_the_comparison(window, verdict) -> None:
    control = _receipt(armed=False)
    control["lanes"][1]["tok_s_median"] = 0
    out = verdict(control, _receipt(armed=True), _smoke())
    assert out["verdict"] == "INVALID"
    assert any("median" in problem for problem in out["gate_failures"])


# --------------------------------------------------------------------------
# the serving smoke's tool-call validation
# --------------------------------------------------------------------------
def _call(name: str, args: object) -> dict:
    raw = args if isinstance(args, str) else json.dumps(args)
    return {"function": {"name": name, "arguments": raw}}


def test_valid_calls_are_accepted(smoke) -> None:
    ok, detail = smoke.validate_call(
        _call("get_weather", {"city": "Lisbon", "units": "celsius"})
    )
    assert ok, detail
    ok, detail = smoke.validate_call(_call("add_numbers", {"a": 4217, "b": 1938}))
    assert ok, detail
    # a qualified city name still names the requested city
    ok, detail = smoke.validate_call(
        _call("get_weather", {"city": "Lisbon, Portugal", "units": "celsius"})
    )
    assert ok, detail
    # addition commutes, so either operand order answers the request
    ok, detail = smoke.validate_call(_call("add_numbers", {"a": 1938, "b": 4217}))
    assert ok, detail


def test_a_swapped_function_is_rejected(smoke) -> None:
    """Both tools are offered, so a valid call for the other one is still wrong."""
    ok, detail = smoke.validate_call(
        _call("add_numbers", {"a": 4217, "b": 1938}), "get_weather"
    )
    assert not ok
    assert any("expected the 'get_weather' call" in line for line in detail)


def _response(*calls: dict) -> dict:
    return {"choices": [{"message": {"content": "", "tool_calls": list(calls)}}]}


def test_one_associates_the_answer_with_the_request(smoke, monkeypatch) -> None:
    """A weather request answered by the addition tool must not pass."""
    monkeypatch.setattr(
        smoke,
        "_post",
        lambda *a, **k: _response(_call("add_numbers", {"a": 4217, "b": 1938})),
    )
    record = smoke._one("tool-weather", "weather?", "get_weather")
    assert record["ok"] is False
    assert record["tool_args_ok"] is False
    assert any(
        "expected the 'get_weather' call" in line for line in record["tool_arg_detail"]
    )

    # the correct function still passes through the same path
    monkeypatch.setattr(
        smoke,
        "_post",
        lambda *a, **k: _response(
            _call("get_weather", {"city": "Lisbon", "units": "celsius"})
        ),
    )
    record = smoke._one("tool-weather", "weather?", "get_weather")
    assert record["ok"] is True


def test_one_requires_exactly_one_call(smoke, monkeypatch) -> None:
    call = _call("get_weather", {"city": "Lisbon", "units": "celsius"})
    for calls in ((), (call, call)):
        monkeypatch.setattr(smoke, "_post", lambda *a, _c=calls, **k: _response(*_c))
        record = smoke._one("tool-weather", "weather?", "get_weather")
        assert record["ok"] is False, calls
        assert any("exactly 1 tool call" in line for line in record["tool_arg_detail"])


def test_a_plain_request_accepts_an_answer_without_calls(smoke, monkeypatch) -> None:
    monkeypatch.setattr(
        smoke, "_post", lambda *a, **k: {"choices": [{"message": {"content": "READY"}}]}
    )
    record = smoke._one("plain-1", "say READY", None)
    assert record["ok"] is True
    assert record["tool_args_ok"] is None


def test_a_plain_request_rejects_an_invented_tool_call(smoke, monkeypatch) -> None:
    """No tools were offered, so any call in the answer is invented."""
    monkeypatch.setattr(
        smoke,
        "_post",
        lambda *a, **k: {
            "choices": [
                {
                    "message": {
                        "content": "READY",
                        "tool_calls": [_call("add_numbers", {"a": 4217, "b": 1938})],
                    }
                }
            ]
        },
    )
    record = smoke._one("plain-1", "say READY", None)
    assert record["ok"] is False
    assert record["n_tool_calls"] == 1
    assert any("no tools were offered" in line for line in record["tool_arg_detail"])


def test_a_plain_request_still_needs_an_answer(smoke, monkeypatch) -> None:
    monkeypatch.setattr(
        smoke, "_post", lambda *a, **k: {"choices": [{"message": {"content": "  "}}]}
    )
    assert smoke._one("plain-1", "say READY", None)["ok"] is False


def test_undeclared_function_is_rejected(smoke) -> None:
    ok, detail = smoke.validate_call(_call("get_forecast", {"city": "Lisbon"}))
    assert not ok
    assert any("undeclared function" in line for line in detail)


def test_extra_property_is_rejected(smoke) -> None:
    ok, detail = smoke.validate_call(
        _call("get_weather", {"city": "Lisbon", "units": "celsius", "days": 3})
    )
    assert not ok
    assert any("undeclared properties" in line for line in detail)


def test_missing_property_is_rejected(smoke) -> None:
    ok, detail = smoke.validate_call(_call("get_weather", {"city": "Lisbon"}))
    assert not ok
    assert any("missing" in line for line in detail)


def test_wrong_argument_values_are_rejected(smoke) -> None:
    ok, detail = smoke.validate_call(
        _call("get_weather", {"city": "Paris", "units": "fahrenheit"})
    )
    assert not ok
    assert any("requested city" in line for line in detail)
    assert any("requested 'celsius'" in line for line in detail)


def test_wrong_operands_are_rejected(smoke) -> None:
    ok, detail = smoke.validate_call(_call("add_numbers", {"a": 4217, "b": 1939}))
    assert not ok
    assert any("do not add the requested" in line for line in detail)


def test_boolean_operands_are_rejected(smoke) -> None:
    """bool is an int subclass, so a naive isinstance check would pass this."""
    ok, detail = smoke.validate_call(_call("add_numbers", {"a": True, "b": 1938}))
    assert not ok
    assert any("must be an integer" in line for line in detail)


def test_non_string_city_is_rejected(smoke) -> None:
    ok, detail = smoke.validate_call(
        _call("get_weather", {"city": 7, "units": "celsius"})
    )
    assert not ok
    assert any("non-empty string" in line for line in detail)


def test_unparseable_and_non_object_arguments_are_rejected(smoke) -> None:
    ok, detail = smoke.validate_call(_call("get_weather", "{not json"))
    assert not ok
    assert any("unparseable" in line for line in detail)
    ok, detail = smoke.validate_call(_call("add_numbers", [4217, 1938]))
    assert not ok
    assert any("not an object" in line for line in detail)


def test_enum_outside_the_declared_set_is_rejected(smoke) -> None:
    ok, detail = smoke.validate_call(
        _call("get_weather", {"city": "Lisbon", "units": "kelvin"})
    )
    assert not ok
    assert any("declared enum" in line for line in detail)


# --------------------------------------------------------------------------
# the serving smoke's log collection
# --------------------------------------------------------------------------
def test_unavailable_logs_are_a_failure(smoke, monkeypatch) -> None:
    """An unreachable worker must not read as a clean scan."""
    done = subprocess.CompletedProcess(
        args=[], returncode=255, stdout="", stderr="ssh: connect: Connection refused\n"
    )
    monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **k: done)
    text, failure = smoke._logs("glm53-exl3-worker", "nvidia@192.168.177.11", 0.0)
    assert text == ""
    assert "255" in failure and "Connection refused" in failure


def test_missing_container_is_a_failure(smoke, monkeypatch) -> None:
    done = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout="",
        stderr="Error: No such container: glm53-exl3-head\n",
    )
    monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **k: done)
    text, failure = smoke._logs("glm53-exl3-head", "local", 0.0)
    assert text == ""
    assert "No such container" in failure


def test_available_logs_are_returned(smoke, monkeypatch) -> None:
    done = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="INFO serving\n", stderr=""
    )
    monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **k: done)
    text, failure = smoke._logs("glm53-exl3-head", "local", 0.0)
    assert text == "INFO serving\n"
    assert failure == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
