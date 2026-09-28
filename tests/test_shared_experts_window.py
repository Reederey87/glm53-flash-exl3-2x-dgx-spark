"""Guards for the window runner and the serving smoke.

Both scripts decide whether a candidate is adopted, so their negative paths
matter as much as their happy path: a comparison that reports ADOPT over a
failed, unhealthy, incoherent, unarmed or memory-starved probe is worse than one
that reports nothing, and a smoke that accepts a malformed tool call is not
serving-correctness evidence.
"""
from __future__ import annotations

import copy
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
        "marker_counts": {"head": 3, "worker": 3} if armed else {"head": 0, "worker": 0},
        "memfree": {
            "head": {"min_kib": 4 * 1024 * 1024},
            "worker": {"min_kib": 4 * 1024 * 1024},
        },
        "lanes": lanes,
    }


def _write(tmp: Path, name: str, payload: dict) -> str:
    path = tmp / name
    path.write_text(json.dumps(payload))
    return str(path)


def _verdict(window, tmp: Path, control: dict, armed: dict) -> dict:
    args = types.SimpleNamespace(
        compare=[_write(tmp, "c.json", control), _write(tmp, "a.json", armed)],
        out=None,
    )
    assert window.compare(args) == 0
    # compare() prints; re-derive the verdict from the same helpers so the test
    # asserts on the logic rather than on stdout formatting.
    problems = window.gate_failures(control, armed)
    return {"problems": problems, "invalid": bool(problems)}


def test_clean_probe_has_no_gate_failures(window) -> None:
    assert window.gate_failures(_receipt(armed=False), _receipt(armed=True)) == []


def test_improvement_is_adoptable(window) -> None:
    control = _receipt(armed=False)
    armed = _receipt(armed=True, scale=1.03)
    assert window.gate_failures(control, armed) == []
    rows, verdict = window.compare_rows(control, armed)
    assert verdict == "ADOPT"
    assert all(row["parity"] and row["above_floor"] for row in rows)


def _fails(window, control: dict, armed: dict) -> bool:
    return bool(window.gate_failures(control, armed))


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
    assert window.gate_failures(control, armed) == []
    _rows, verdict = window.compare_rows(control, armed)
    assert verdict == "REVERT"


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
    assert any("requested 1938" in line for line in detail)


def test_boolean_operands_are_rejected(smoke) -> None:
    """bool is an int subclass, so a naive isinstance check would pass this."""
    ok, detail = smoke.validate_call(_call("add_numbers", {"a": True, "b": 1938}))
    assert not ok
    assert any("must be an integer" in line for line in detail)


def test_non_string_city_is_rejected(smoke) -> None:
    ok, detail = smoke.validate_call(_call("get_weather", {"city": 7, "units": "celsius"}))
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
