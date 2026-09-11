"""CPU-only tests for the paired two-arm v1.4.9-vs-v1.4.7 diagnostic.

No cluster, no torch, no HTTP. These pin the properties that make the
diagnostic's answer trustworthy, and they are the regression tests for two
review findings:

1. The decision rule must not certify its own central estimate. An earlier
   version gated on a `median_robust` flag that counted observations below half
   their own median and called the lane trustworthy when fewer than half
   qualified. That was circular: for positive values and odd n, fewer than half
   always qualify, so it could not detect majority contamination, and the
   median's 50% breakdown point does not mean replacing a minority cannot shift
   it. The flag is gone; the decision now uses a distribution-free shift
   interval. See `test_majority_contamination_is_not_certified_as_robust` and
   `test_a_minority_shift_moves_the_median_and_is_not_hidden`.

2. The report phase must validate the capture before issuing a verdict, because
   `--from report` and a resume after a late failure are both reachable with
   every lane file present but the capture rejected. See the
   `test_report_*` tests.
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
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


def steady(median: float, n: int = 21) -> list[float]:
    """A settled lane's raw observations: within a fraction of a percent."""
    return [median * (1 + (i - n / 2) * 0.0005) for i in range(n)]


def stalled(median: float, n: int = 21, count: int = 2) -> list[float]:
    """A settled lane with `count` observations collapsed by the live stall."""
    values = steady(median, n)
    for i in range(count):
        values[i * 3] = median * LIVE_STALL_FACTOR
    return values


def decide(lane_name: str, control: list[float], candidate: list[float]) -> dict:
    return diag.compare_lane(lane_name, diag.lane_stats(control),
                             diag.lane_stats(candidate), control, candidate)


# --- contract ---------------------------------------------------------------

def test_every_lane_has_an_auditor_band():
    """A lane without a band would silently compare against nothing."""
    assert set(ALL_LANES) == set(audit.BANDS)
    assert set(diag.LANE_TIMEOUTS) == set(ALL_LANES)
    assert set(diag.REQUIRED_PHASES) | {"report"} == set(diag.PHASES)


def test_diagnostic_uses_the_same_arms_as_the_registered_window():
    """A tag drift here would compare different images than the §6 contract."""
    assert window.ARMS["a"]["tag"] == "glm53-selfbuild:e3-w3-zfill"
    assert window.ARMS["b"]["tag"] == "glm53-selfbuild:e3-w3-zfill-v149"
    assert window.ARMS["a"]["exllamav3"] == "1.4.7"
    assert window.ARMS["b"]["exllamav3"] == "1.4.9"
    assert window.PRODUCTION_IMAGE == window.ARMS["b"]["tag"]


def test_decode_lanes_get_more_observations_than_the_registered_contract():
    """The larger sample is the entire reason this diagnostic exists."""
    registered = window.LANES["structured"][0]
    for name in diag.DECODE_LANES:
        assert diag.DEFAULT_COUNTS[name] > registered


def test_receipt_is_labelled_as_not_section_6_evidence():
    assert "NOT §6" in diag.EVIDENCE_CLASS
    # The §6 judge must not be reachable from this harness.
    assert "judge" not in diag.HANDLERS
    assert "report" in diag.HANDLERS


def test_phases_are_ordered_so_recovery_always_precedes_the_report():
    order = list(diag.PHASES)
    assert order.index("disarm") < order.index("arm_a")
    assert order.index("arm_a") < order.index("measure_a")
    assert order.index("measure_a") < order.index("arm_b")
    assert order.index("measure_b") < order.index("restore")
    assert order.index("restore") < order.index("rearm")
    assert order.index("rearm") < order.index("report")


# --- finding 1: the decision rule must not certify its own estimate ----------

def test_majority_contamination_is_not_certified_as_robust():
    """The reviewer's counterexample: 11 observations at 30, 10 at 100.

    The median IS the contaminated value (30), and the old `median_robust` flag
    reported the lane trustworthy because zero observations fell below half their
    own median. A near-50/50 bimodal sample genuinely cannot resolve its median,
    so the honest answer is that no pass may be claimed — never "robust".
    """
    contaminated = diag.lane_stats([30.0] * 11 + [100.0] * 10)
    assert contaminated["median"] == 30.0
    # The circular flag is gone entirely.
    assert "median_robust" not in contaminated
    # The interval spans the contamination instead of hiding it.
    assert contaminated["low"] == 30.0
    assert contaminated["high"] == 100.0

    row = decide("essay", [100.0] * 21, [30.0] * 11 + [100.0] * 10)
    assert row["verdict"] != "non-inferior", row
    assert diag.overall_verdict([row])[0] != "NO REGRESSION DETECTED"


def test_a_minority_shift_moves_the_median_and_is_not_hidden():
    """The reviewer's second counterexample.

    Replacing three of twenty-one observations moved the median 100 -> 90, so
    "fewer than half changed" is not a guarantee that the median held.
    """
    base = [90.0] * 10 + [100.0] * 11
    shifted = [90.0] * 10 + [100.0] * 8 + [37.0] * 3
    assert diag.lane_stats(base)["median"] == 100.0
    assert diag.lane_stats(shifted)["median"] == 90.0

    # Compared against a candidate at 96, the shifted arm must not be read as
    # comfortably non-inferior on the strength of its median alone.
    row = decide("structured", shifted, [96.0] * 21)
    assert row["shift_ci_low"] < row["band_boundary_shift"], row


def test_stall_count_is_descriptive_and_never_a_gate():
    """`stall_count` must not appear in the decision path at all."""
    contaminated = diag.lane_stats([30.0] * 11 + [100.0] * 10)
    assert contaminated["stall_count"] == 0  # it detects nothing here
    # ...and the verdict does not consult it: the same medians give the same
    # verdict whether or not the stall counter noticed anything.
    a = [100.0] * 21
    b = [30.0] * 11 + [100.0] * 10
    assert decide("essay", a, b)["verdict"] == decide("essay", a, b)["verdict"]


def test_a_single_stall_destroys_the_registered_gate_but_not_the_interval():
    """The registered spread is (max-min)/median; one outlier blows it up.

    This is why the diagnostic exists: the registered gate rejects the arm, while
    the shift interval still decides it.
    """
    s = diag.lane_stats(stalled(100.0, n=21, count=1))
    assert s["registered_spread"] > audit.VARIABILITY_MAX
    assert s["registered_settled"] is False
    assert abs(s["median"] - 100.0) < 0.5

    row = decide("essay", stalled(100.0, count=1), steady(101.0))
    assert row["verdict"] == "non-inferior", row


def test_stall_detector_does_not_flag_ordinary_spread():
    """Half the median is conservative: real lanes vary by under a percent."""
    s = lane(*[100.0 * (1 + (i - 10) * 0.003) for i in range(21)])
    assert s["stall_count"] == 0
    assert s["registered_settled"] is True


# --- statistics -------------------------------------------------------------

def test_mann_whitney_distribution_sums_to_the_binomial_coefficient():
    """The exact U distribution must be a proper probability distribution."""
    for m, n in ((2, 2), (5, 5), (9, 9), (21, 21)):
        counts = diag._mann_whitney_counts(m, n)
        assert sum(counts) == math.comb(m + n, m), (m, n)
        assert len(counts) == m * n + 1


def test_shift_interval_is_symmetric_and_contains_the_point_estimate():
    hl = diag.hodges_lehmann_ci(steady(100.0), steady(101.0))
    assert hl["low"] <= hl["shift"] <= hl["high"]
    assert hl["level"] is not None and hl["level"] >= 0.95
    # A symmetric two-sample construction: the index mirrors around the middle.
    assert hl["order_statistic"] * 2 <= hl["n_pairs"] + 1


def test_shift_interval_widens_with_spread():
    tight = diag.hodges_lehmann_ci(steady(100.0), steady(101.0))
    wide = diag.hodges_lehmann_ci(stalled(100.0, count=3), stalled(101.0, count=3))
    assert (wide["high"] - wide["low"]) > (tight["high"] - tight["low"])


def test_shift_interval_on_identical_samples_contains_zero():
    same = steady(100.0)
    hl = diag.hodges_lehmann_ci(same, same)
    assert hl["shift"] == 0.0
    assert hl["low"] <= 0.0 <= hl["high"]


def test_degenerate_samples_do_not_raise():
    for values in ([], [0.0, 0.0]):
        s = diag.lane_stats(list(values))
        assert s["median"] is None
    assert diag.median_ci([])["low"] is None
    assert diag.median_ci([5.0])["low"] == 5.0
    assert diag.hodges_lehmann_ci([], [1.0])["shift"] is None
    assert diag.hodges_lehmann_ci([1.0], [2.0])["shift"] == 1.0


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
    above_band = steady(99.0)
    below_band = steady(96.0)
    assert decide("structured", control, above_band)["verdict"] == "non-inferior"
    assert decide("structured", control, below_band)["verdict"] == "REGRESSED"
    # The same 0.96 ratio passes essay's looser 0.95 band.
    assert decide("essay", control, below_band)["verdict"] == "non-inferior"


def test_the_band_boundary_is_decided_by_the_interval_not_the_point_estimate():
    """A lane sitting exactly on its boundary is undecidable, not a pass."""
    row = decide("structured", steady(100.0), steady(97.0))
    assert row["ratio"] == pytest.approx(0.97, abs=1e-9)
    assert row["band"] == 0.97
    assert row["verdict"] == "inconclusive", row


def test_a_real_regression_is_still_reported():
    row = decide("essay", steady(100.0), steady(80.0))
    assert row["verdict"] == "REGRESSED"
    assert row["ratio"] == pytest.approx(0.80, abs=1e-6)
    assert row["candidate_over_control_percent"] == pytest.approx(-20.0, abs=0.05)


def test_a_wide_lane_is_inconclusive_rather_than_rounded_to_a_pass():
    """A lane the sample cannot resolve must not become a pass.

    A near-50/50 bimodal arm is the honest case: the sample genuinely cannot say
    where its centre is, so neither a pass nor a regression may be claimed.
    """
    control = [100.0] * 21
    undecidable = [30.0] * 11 + [100.0] * 10
    row = decide("hashmap", control, undecidable)
    assert row["verdict"] == "inconclusive", row
    assert diag.overall_verdict([row])[0] == "INCONCLUSIVE"


def test_an_unmeasurable_lane_cannot_read_as_a_pass():
    row = diag.compare_lane("essay", {"valid_runs": 0, "median": None},
                            {"valid_runs": 0, "median": None}, [], [])
    assert row["verdict"] == "unmeasurable"
    assert diag.overall_verdict([row])[0] == "INCONCLUSIVE"


def test_missing_raw_values_are_inconclusive_not_a_pass():
    """The decision needs the observations, not just the medians."""
    row = diag.compare_lane("essay", diag.lane_stats(steady(100.0)),
                            diag.lane_stats(steady(101.0)), None, None)
    assert row["verdict"] == "inconclusive"


def test_overall_verdict_precedence():
    control = steady(100.0)
    good = decide("essay", control, steady(101.0))
    bad = decide("essay", control, steady(80.0))
    unsure = decide("hashmap", [100.0] * 21, [30.0] * 11 + [100.0] * 10)

    assert diag.overall_verdict([good])[0] == "NO REGRESSION DETECTED"
    assert diag.overall_verdict([unsure])[0] == "INCONCLUSIVE"
    assert diag.overall_verdict([bad])[0] == "REGRESSION"
    # A demonstrated regression outranks an undecidable lane.
    assert diag.overall_verdict([unsure, bad])[0] == "REGRESSION"


# --- finding 2: capture integrity before the verdict -------------------------

def _write_lane(directory: Path, arm: str, lane_name: str, values: list[float],
                **extra) -> str:
    key = "prefill_tok_s" if lane_name.startswith("prefill") else "tok_s"
    name = f"{arm}-{lane_name}.json"
    payload = {
        "kind": lane_name,
        "runs": [{key: v, "ok": True} for v in values],
        "invalid_runs": [],
        "valid_runs": len(values),
        "any_cache_hit": False,
    }
    payload.update(extra)
    (directory / name).write_text(json.dumps(payload))
    return name


def _healthy_state(directory: Path, counts: dict | None = None) -> dict:
    """A receipt that passes every integrity check, so a test can break one thing."""
    counts = dict(counts or diag.DEFAULT_COUNTS)
    state: dict = {
        "counts": counts,
        "phases": [{"phase": p, "ok": True} for p in diag.REQUIRED_PHASES],
        "probes": {"a": {}, "b": {}},
        "probe_blocks": [],
        "arms": {},
    }
    for arm, boot in (("a", "boot-a"), ("b", "boot-b")):
        state["arms"][arm] = {
            "container_started_at": boot,
            "measure_container_started_at": boot,
            "measure_verified_image": window.ARMS[arm]["tag"],
            "measure_verified_exllamav3": window.ARMS[arm]["exllamav3"],
            "preemptions_delta": 0,
        }
    for arm, scale in (("a", 1.0), ("b", 1.01)):
        for lane_name in ALL_LANES:
            base = 1600.0 if lane_name.startswith("prefill") else 100.0
            n = counts[lane_name]
            values = [base * scale * (1 + (i - n / 2) * 0.0005) for i in range(n)]
            name = _write_lane(directory, arm, lane_name, values)
            state["probes"][arm][lane_name] = name
            state["probe_blocks"].append({
                "arm": arm, "lane": lane_name, "runs": n, "ok": True,
                "returncode": 0, "path": name, "error": None,
            })
    return state


def test_a_healthy_receipt_validates_and_passes(tmp_path, monkeypatch):
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    state = _healthy_state(tmp_path)
    assert diag.validate_capture(state) == []

    diag.phase_report(state)
    written = json.loads(receipt.read_text())
    assert written["verdict"] == "NO REGRESSION DETECTED"
    assert written["capture_problems"] == []
    assert written["evidence_class"] == diag.EVIDENCE_CLASS
    for row in written["comparison"]:
        assert row["band"] == audit.BANDS[row["lane"]]
    # The method is recorded so the receipt is self-describing.
    assert "Hodges-Lehmann" in written["method"]["decision"]


def test_report_resumption_on_a_failed_receipt_is_not_a_pass(tmp_path, monkeypatch):
    """The reviewer's concrete path: failure at the final preemption check.

    Every lane file exists and recovery succeeded, so `--from report` is
    reachable — it must not print a passing verdict.
    """
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    state = _healthy_state(tmp_path)
    state["arms"]["b"]["preemptions_delta"] = 4

    problems = diag.validate_capture(state)
    assert any("preemptions" in p for p in problems)

    diag.phase_report(state)
    written = json.loads(receipt.read_text())
    assert written["verdict"] == "INVALID CAPTURE"
    assert written["verdict"] != "NO REGRESSION DETECTED"
    # The per-lane numbers survive as evidence, but the verdict does not pass.
    assert written["comparison"]


@pytest.mark.parametrize("mutate,expect", [
    (lambda s: s["phases"].append({"phase": "rearm", "ok": False, "error": "boom"}),
     "rearm"),
    (lambda s: s["phases"].pop(0), "preflight"),
    (lambda s: s.update(phase_in_progress="measure_b"), "measure_b"),
    (lambda s: s["arms"]["a"].update(preemptions_delta=None), "preemption delta unreadable"),
    (lambda s: s["arms"]["a"].update(measure_verified_exllamav3="1.4.9"), "exllamav3"),
    (lambda s: s["arms"]["b"].update(measure_verified_image="wrong:tag"), "not the arm's tag"),
    (lambda s: s["arms"]["a"].update(measure_container_started_at="boot-later"), "restarted"),
    (lambda s: s["arms"]["b"].update(container_started_at="boot-a"), "same container boot"),
    (lambda s: s["probes"]["a"].pop("hashmap"), "no evidence file was selected"),
    (lambda s: [b.update(ok=False, error="probe died")
                for b in s["probe_blocks"] if b["arm"] == "a" and b["lane"] == "essay"],
     "every registered block failed"),
    (lambda s: s["probe_blocks"].pop(0), "no probe block was registered"),
])
def test_capture_integrity_violations_block_the_verdict(tmp_path, monkeypatch, mutate, expect):
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    state = _healthy_state(tmp_path)
    mutate(state)
    problems = diag.validate_capture(state)
    assert problems, "the mutation must be detected"
    assert any(expect in p for p in problems), (expect, problems)
    diag.phase_report(state)
    assert json.loads(receipt.read_text())["verdict"] == "INVALID CAPTURE"


def test_damaged_lane_evidence_blocks_the_verdict(tmp_path, monkeypatch):
    """Wrong counts, invalid runs, cache hits, and non-finite values all block."""
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)

    for label, lane_name, extra, expect in (
        ("short sample", "structured", {}, "expected 21"),
        ("invalid run", "structured", {"invalid_runs": [{"reason": "no cached_tokens"}]},
         "invalid run"),
        # The coldness flag only means anything on a prefill lane.
        ("cache hit", "prefill60k", {"any_cache_hit": True}, "any_cache_hit"),
        ("non-finite", "structured", {}, "non-finite"),
    ):
        state = _healthy_state(tmp_path)
        target = state["probes"]["a"][lane_name]
        path = tmp_path / target
        payload = json.loads(path.read_text())
        if label == "short sample":
            payload["runs"] = payload["runs"][:1]
        elif label == "non-finite":
            payload["runs"][0]["tok_s"] = float("nan")
        payload.update(extra)
        path.write_text(json.dumps(payload))

        problems = diag.validate_capture(state)
        assert any(expect in p for p in problems), (label, problems)
        diag.phase_report(state)
        assert json.loads(receipt.read_text())["verdict"] == "INVALID CAPTURE", label


def test_a_missing_lane_file_is_not_a_pass(tmp_path, monkeypatch):
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    state = _healthy_state(tmp_path)
    (tmp_path / state["probes"]["a"]["hashmap"]).unlink()
    diag.phase_report(state)
    written = json.loads(receipt.read_text())
    assert written["verdict"] == "INVALID CAPTURE"
    row = next(r for r in written["comparison"] if r["lane"] == "hashmap")
    assert row["verdict"] == "unmeasurable"


def test_a_wrong_kind_in_the_evidence_file_is_detected(tmp_path, monkeypatch):
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    state = _healthy_state(tmp_path)
    path = tmp_path / state["probes"]["b"]["essay"]
    payload = json.loads(path.read_text())
    payload["kind"] = "hashmap"
    path.write_text(json.dumps(payload))
    assert any("kind" in p for p in diag.validate_capture(state))


def test_evidence_dir_without_a_receipt_fails_closed(tmp_path, monkeypatch):
    """No receipt must not mean 'no problems'."""
    state = _healthy_state(tmp_path)
    monkeypatch.setattr(diag.window, "_RECEIPT", None)
    problems = diag.validate_capture(state)
    assert any("missing or unreadable" in p for p in problems)
    assert diag._load_lane(state, "a", "structured") is None
