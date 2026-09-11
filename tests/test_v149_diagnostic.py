"""CPU-only tests for the paired two-arm v1.4.9-vs-v1.4.7 diagnostic.

No cluster, no torch, no HTTP. These pin the properties that make the
diagnostic's answer trustworthy: that the median is used and is defended as
robust, that the per-lane bands come from the auditor rather than being
redeclared, that a real regression is still caught, and that the receipt can
never be mistaken for §6 qualification evidence.

The diagnostic exists because the registered §6 window's variability gate is
`(max - min) / median <= 0.30`, which one stall destroys. `test_a_single_stall_
destroys_the_registered_gate_but_not_the_median` is the test that records why.
"""

from __future__ import annotations

import importlib.util
import json
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


diag = _load(SCRIPTS / "diagnose_v149_ab.py", "glm53_diag_v149")
audit = _load(SCRIPTS / "audit_v149_qualification.py", "glm53_audit_v149")
window = _load(SCRIPTS / "run_v149_qualification_window.py", "glm53_window_v149")

ALL_LANES = (*diag.DECODE_LANES, *diag.PREFILL_LANES)
# The live host's stall: one observation in twenty collapses by roughly this.
LIVE_STALL_FACTOR = 1 / 2.7


def lane(*values: float) -> dict:
    return diag.lane_stats(list(values))


def steady(median: float, n: int = 21) -> dict:
    """A settled lane: n observations within a fraction of a percent."""
    return lane(*[median * (1 + (i - n / 2) * 0.0005) for i in range(n)])


def stalled(median: float, n: int = 21, count: int = 2) -> dict:
    """A settled lane with `count` observations collapsed by the live stall."""
    values = [median * (1 + (i - n / 2) * 0.0005) for i in range(n)]
    for i in range(count):
        values[i * 3] = median * LIVE_STALL_FACTOR
    return lane(*values)


# --- contract ---------------------------------------------------------------

def test_every_lane_has_an_auditor_band():
    """A lane without a band would silently compare against nothing."""
    assert set(ALL_LANES) == set(audit.BANDS)
    assert set(diag.LANE_TIMEOUTS) == set(ALL_LANES)


def test_diagnostic_uses_the_same_arms_as_the_registered_window():
    """A tag drift here would compare different images than the §6 contract."""
    assert window.ARMS["a"]["tag"] == "glm53-selfbuild:e3-w3-zfill"
    assert window.ARMS["b"]["tag"] == "glm53-selfbuild:e3-w3-zfill-v149"
    assert window.ARMS["a"]["exllamav3"] == "1.4.7"
    assert window.ARMS["b"]["exllamav3"] == "1.4.9"
    # Production is arm B, so `phase_restore`'s verification is the right one.
    assert window.PRODUCTION_IMAGE == window.ARMS["b"]["tag"]


def test_decode_lanes_get_more_observations_than_the_registered_contract():
    """The larger sample is the entire reason this diagnostic exists."""
    registered = window.LANES["structured"][0]
    for name in diag.DECODE_LANES:
        assert diag.DEFAULT_COUNTS[name] > registered


def test_receipt_is_labelled_as_not_section_6_evidence():
    assert "NOT §6" in diag.EVIDENCE_CLASS
    assert diag.EVIDENCE_CLASS != ""

    def handler_names():
        return set(diag.HANDLERS)

    # The §6 phases the diagnostic must NOT claim to run.
    assert "judge" not in handler_names()
    assert "report" in handler_names()


def test_phases_are_ordered_so_recovery_always_precedes_the_report():
    order = list(diag.PHASES)
    assert order.index("disarm") < order.index("arm_a")
    assert order.index("arm_a") < order.index("measure_a")
    assert order.index("measure_a") < order.index("arm_b")
    assert order.index("measure_b") < order.index("restore")
    assert order.index("restore") < order.index("rearm")
    # The report reads the receipts, so it must come after both arms.
    assert order.index("rearm") < order.index("report")


# --- the reason the diagnostic exists ---------------------------------------

def test_a_single_stall_destroys_the_registered_gate_but_not_the_median():
    """The registered spread is (max-min)/median; one outlier blows it up.

    This is the observation that made the registered window undecidable, and the
    property the diagnostic's median comparison is chosen to survive.
    """
    s = stalled(100.0, n=21, count=1)
    assert s["stall_count"] == 1
    assert s["median_robust"] is True
    # The registered gate would have rejected this arm as unsettled...
    assert s["registered_spread"] > audit.VARIABILITY_MAX
    assert s["registered_settled"] is False
    # ...while the median is essentially unmoved by the stall.
    assert abs(s["median"] - 100.0) < 0.5
    assert s["mad_over_median"] < 0.01


def test_median_survives_the_live_stall_rate_but_is_not_assumed_to():
    s = stalled(100.0, n=21, count=2)
    assert s["stall_count"] == 2
    assert s["median_robust"] is True
    assert s["registered_settled"] is False

    # Past half the sample the median is no longer a central estimate, and the
    # diagnostic must say so rather than report a confident number.
    half = lane(100.0, 30.0, 100.0, 30.0)
    assert half["median_robust"] is False
    assert diag.compare_lane("essay", steady(100.0), half)["verdict"] == "inconclusive"


def test_stall_detector_does_not_flag_ordinary_spread():
    """Half the median is conservative: real lanes vary by under a percent."""
    s = lane(*[100.0 * (1 + (i - 10) * 0.003) for i in range(21)])
    assert s["stall_count"] == 0
    assert s["registered_settled"] is True


# --- statistics -------------------------------------------------------------

def test_median_and_spread_are_computed_over_valid_runs_only():
    s = lane(100.0, 102.0, 98.0)
    assert s["median"] == 100.0
    assert s["valid_runs"] == 3
    assert s["min"] == 98.0 and s["max"] == 102.0


def test_empty_or_zero_lanes_report_no_median_instead_of_raising():
    for values in ([], [0.0, 0.0]):
        s = diag.lane_stats(list(values))
        assert s["median"] is None
        assert s["median_robust"] is False


def test_run_values_reads_the_lane_specific_key_and_skips_nulls():
    decode = {"runs": [{"tok_s": 64.0}, {"tok_s": None}, {"tok_s": 65.0}]}
    assert diag.run_values("structured", decode) == [64.0, 65.0]
    prefill = {"runs": [{"prefill_tok_s": 1600.0}, {"prefill_tok_s": 1610.0}]}
    assert diag.run_values("prefill60k", prefill) == [1600.0, 1610.0]
    # A decode lane must not read a prefill key, or vice versa.
    assert diag.run_values("structured", prefill) == []
    assert diag.run_values("prefill60k", decode) == []


# --- verdicts ---------------------------------------------------------------

def test_non_inferiority_uses_each_lane_s_own_band():
    control = steady(100.0)
    # 97.2/100.2... use an exact ratio so the boundary is unambiguous.
    at_band = steady(97.0)
    below_band = steady(96.0)
    assert diag.compare_lane("structured", control, at_band)["verdict"] == "non-inferior"
    assert diag.compare_lane("structured", control, below_band)["verdict"] == "REGRESSED"
    # The same 0.96 ratio passes essay's looser 0.95 band.
    assert diag.compare_lane("essay", control, below_band)["verdict"] == "non-inferior"


def test_a_real_regression_is_still_reported():
    row = diag.compare_lane("essay", steady(100.0), steady(80.0))
    assert row["verdict"] == "REGRESSED"
    assert row["ratio"] == pytest.approx(0.80, abs=1e-6)
    assert row["candidate_over_control_percent"] == pytest.approx(-20.0, abs=0.05)


def test_an_unmeasurable_lane_cannot_read_as_a_pass():
    row = diag.compare_lane("essay", steady(100.0), {"valid_runs": 0, "median": None})
    assert row["verdict"] == "unmeasurable"
    assert diag.overall_verdict([row])[0] == "INCONCLUSIVE"


def test_overall_verdict_precedence():
    control = steady(100.0)
    good = diag.compare_lane("essay", control, steady(101.0))
    bad = diag.compare_lane("essay", control, steady(80.0))
    unsure = diag.compare_lane("essay", control, lane(100.0, 30.0, 100.0, 30.0))

    assert diag.overall_verdict([good])[0] == "NO REGRESSION DETECTED"
    assert diag.overall_verdict([unsure])[0] == "INCONCLUSIVE"
    assert diag.overall_verdict([bad])[0] == "REGRESSION"
    # A demonstrated regression outranks an undecidable lane.
    assert diag.overall_verdict([unsure, bad])[0] == "REGRESSION"


# --- report -----------------------------------------------------------------

def _write_lane(directory: Path, arm: str, lane_name: str, values: list[float]) -> str:
    key = "prefill_tok_s" if lane_name.startswith("prefill") else "tok_s"
    name = f"{arm}-{lane_name}.json"
    (directory / name).write_text(json.dumps({
        "kind": lane_name,
        "runs": [{key: v, "ok": True} for v in values],
        "invalid_runs": [],
        "valid_runs": len(values),
    }))
    return name


def test_report_writes_a_receipt_labelled_diagnostic(tmp_path, monkeypatch):
    """End-to-end over synthetic probe outputs: the receipt carries the verdict
    and the evidence class, and never a §6 claim."""
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    state = {
        "counts": dict(diag.DEFAULT_COUNTS),
        "probes": {"a": {}, "b": {}},
    }
    for arm, scale in (("a", 1.0), ("b", 1.01)):
        for lane_name in ALL_LANES:
            base = 1600.0 if lane_name.startswith("prefill") else 100.0
            values = [base * scale * (1 + (i - 5) * 0.001) for i in range(11)]
            # Arm A's essay lane carries one live-scale stall.
            if arm == "a" and lane_name == "essay":
                values[0] = base * scale * LIVE_STALL_FACTOR
            state["probes"][arm][lane_name] = _write_lane(tmp_path, arm, lane_name, values)

    diag.phase_report(state)

    written = json.loads(receipt.read_text())
    assert written["evidence_class"] == diag.EVIDENCE_CLASS
    assert written["verdict"] == "NO REGRESSION DETECTED"
    assert set(written["stats"]) == set(ALL_LANES)
    # The stall on arm A's essay lane is recorded, and did not move the median.
    assert written["stats"]["essay"]["a"]["stall_count"] == 1
    assert written["stats"]["essay"]["a"]["median_robust"] is True
    assert written["stats"]["essay"]["a"]["registered_settled"] is False
    # Every comparison row carries its band from the auditor.
    for row in written["comparison"]:
        assert row["band"] == audit.BANDS[row["lane"]]


def test_report_survives_a_missing_lane_receipt(tmp_path, monkeypatch):
    """A lane whose file is gone must be undecidable, not silently skipped."""
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    state = {"counts": dict(diag.DEFAULT_COUNTS), "probes": {"a": {}, "b": {}}}
    for arm in ("a", "b"):
        for lane_name in ALL_LANES:
            if lane_name == "hashmap":
                continue
            base = 1600.0 if lane_name.startswith("prefill") else 100.0
            state["probes"][arm][lane_name] = _write_lane(
                tmp_path, arm, lane_name, [base * (1 + i * 0.001) for i in range(5)]
            )
    diag.phase_report(state)
    written = json.loads(receipt.read_text())
    assert written["verdict"] == "INCONCLUSIVE"
    hashmap_row = next(r for r in written["comparison"] if r["lane"] == "hashmap")
    assert hashmap_row["verdict"] == "unmeasurable"
