"""CPU-only tests for the paired two-arm v1.4.9-vs-v1.4.7 diagnostic.

No cluster, no torch, no HTTP. These pin the properties that make the
diagnostic's answer trustworthy, and they are the regression tests for the
review findings:

1. The decision rule must not certify its own central estimate. An earlier
   version gated on a `median_robust` flag that counted observations below half
   their own median and called the lane trustworthy when fewer than half
   qualified. That was circular: for positive values and odd n, fewer than half
   always qualify, so it could not detect majority contamination, and the
   median's 50% breakdown point does not mean replacing a minority cannot shift
   it. See `test_majority_contamination_is_not_certified_as_robust` and
   `test_a_minority_shift_moves_the_median_and_is_not_hidden`.

2. The decision rule must bound the quantity the bands are written in. A
   Hodges-Lehmann shift estimates the MEDIAN OF PAIRWISE DIFFERENCES, which
   equals the difference of medians only under a location-shift model. Two
   differently shaped arms can have a median ratio below the band while the
   shift interval sits comfortably inside it. The rule now composes the two
   arms' distribution-free median intervals into an interval for the median
   ratio. See `test_a_differing_shape_cannot_pass_on_the_shift_estimand`.

3. A lane whose sample cannot reach the required coverage must be inconclusive
   rather than reported at a level it never achieved. See the
   `test_*coverage*` tests.

4. The report phase must validate the capture before issuing a verdict, because
   `--from report` and a resume after a late failure are both reachable with
   every lane file present but the capture rejected. A retry that eventually
   succeeded must not erase the record of the attempt that failed first. See the
   `test_report_*` and `test_an_interrupted_retry_*` tests.

5. Validation must fail closed on an *incomplete* contract, not only on a
   demonstrably false one. Success has to be affirmative (`ok is True`), every
   lane needs a usable count before its evidence can be measured against one,
   and both nodes' images and boots must be the arm's own — a check that reads
   only the head cannot see a worker running the other arm's build. See
   `test_a_phase_with_no_success_flag_is_not_evidence_of_success` and the
   `worker`/`no usable observation count`/`registered 7 runs` cases.

6. An interruption must survive a resume of the very phase it interrupted.
   Exempting that case let `--from rearm --to report` turn a run killed inside
   `rearm` into NO REGRESSION DETECTED. See
   `test_resuming_the_interrupted_evidence_phase_does_not_launder_it` and
   `test_the_running_phase_is_not_mistaken_for_a_crashed_one`, which also pins
   the narrow exception for the recomputable terminal phase.
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
                             diag.lane_stats(candidate))


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


def test_every_lane_gets_at_least_the_coverage_minimum():
    """A default that cannot reach MIN_LEVEL would be refused at startup anyway."""
    minimum = diag.min_runs_for_level()
    for name, n in diag.DEFAULT_COUNTS.items():
        assert n >= minimum, (name, n)


def test_every_default_count_can_reach_the_required_coverage():
    """The composed coverage a default count buys must clear MIN_LEVEL."""
    for name, n in diag.DEFAULT_COUNTS.items():
        ci = diag.median_ci([float(i) for i in range(n)], level=diag.ARM_LEVEL)
        assert 2.0 * ci["level"] - 1.0 >= diag.MIN_LEVEL, (name, n, ci["level"])


def test_decode_lanes_get_more_observations_than_the_registered_contract():
    """The larger sample is the entire reason this diagnostic exists."""
    registered = window.LANES["structured"][0]
    for name in diag.DECODE_LANES:
        assert diag.DEFAULT_COUNTS[name] > registered
    # The binding lane is the one with the widest genuine spread on this host.
    assert diag.DEFAULT_COUNTS["hashmap"] >= max(
        diag.DEFAULT_COUNTS[name] for name in diag.DECODE_LANES
    )


def test_the_expensive_prefill_lane_is_not_padded_to_the_decode_count():
    """prefill240k costs ~152 s per observation, so its count is set by need."""
    assert diag.DEFAULT_COUNTS["prefill240k"] < diag.DEFAULT_COUNTS["hashmap"]
    # ...but it still buys at least one order statistic, so one stall cannot
    # widen the interval to the extremes.
    ci = diag.median_ci([float(i) for i in range(diag.DEFAULT_COUNTS["prefill240k"])],
                        level=diag.ARM_LEVEL)
    assert ci["order_statistic"] >= 2


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
    # comfortably non-inferior on the strength of its median alone: the point
    # ratio flatters it (1.067), while the interval's lower bound sits below the
    # 0.97 band because the shift widened the control's own interval.
    row = decide("structured", shifted, [96.0] * 21)
    assert row["ratio"] > row["band"], row
    assert row["ratio_ci_low"] < row["band"], row
    assert row["verdict"] == "inconclusive", row


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

def test_median_interval_is_not_circular_and_widens_as_the_sample_shrinks():
    """The interval must come from the sample size, not from the sample's own spread.

    Both samples here are drawn from the same fixed range, so the only thing that
    differs is how many observations pin the median down.
    """
    spread = [100.0 + (i % 7 - 3) * 0.01 for i in range(21)]
    small = diag.median_ci(spread[:7], level=0.975)
    large = diag.median_ci(spread, level=0.975)
    assert (large["high"] - large["low"]) < (small["high"] - small["low"])
    assert small["low"] <= 100.0 <= small["high"]
    assert large["low"] <= 100.0 <= large["high"]


def test_median_interval_level_never_exceeds_what_the_sample_supports():
    """The reported level is the ACHIEVED one, so it cannot overstate the sample."""
    for n in (1, 2, 3, 5, 7, 9, 21):
        ci = diag.median_ci(steady(100.0, n=n), level=0.975)
        if n == 1:
            assert ci["level"] is None
        else:
            assert 0.0 < ci["level"] <= 1.0
    assert diag.median_ci(steady(100.0, n=21), level=0.975)["level"] >= 0.975


def test_min_runs_for_level_is_the_smallest_sample_that_reaches_the_requirement():
    minimum = diag.min_runs_for_level()
    assert minimum == 7
    assert minimum == diag.min_runs_for_level(diag.MIN_LEVEL)
    # One fewer cannot, which is what makes it the minimum.
    for n in range(2, minimum):
        ci = diag.median_ci(steady(100.0, n=n), level=diag.ARM_LEVEL)
        composed = 2.0 * ci["level"] - 1.0 if ci["level"] is not None else 0.0
        assert composed < diag.MIN_LEVEL, n
    for n in range(minimum, minimum + 4):
        ci = diag.median_ci(steady(100.0, n=n), level=diag.ARM_LEVEL)
        assert 2.0 * ci["level"] - 1.0 >= diag.MIN_LEVEL, n


def test_the_composed_interval_is_conservative_for_the_ratio():
    """lo pairs the candidate's low with the control's high, and vice versa."""
    control = diag.lane_stats(steady(100.0))
    candidate = diag.lane_stats(steady(110.0))
    row = diag.compare_lane("essay", control, candidate)
    assert row["ratio_ci_low"] == pytest.approx(candidate["low"] / control["high"])
    assert row["ratio_ci_high"] == pytest.approx(candidate["high"] / control["low"])
    # The composition costs coverage, and the cost is reported rather than hidden.
    assert row["ratio_ci_level"] == pytest.approx(
        2.0 * min(control["level"], candidate["level"]) - 1.0
    )
    assert row["ratio_ci_low"] <= row["ratio"] <= row["ratio_ci_high"]


def test_degenerate_samples_do_not_raise():
    for values in ([], [0.0, 0.0]):
        s = diag.lane_stats(list(values))
        assert s["median"] is None
    assert diag.median_ci([])["low"] is None
    assert diag.median_ci([5.0])["low"] == 5.0
    assert diag.median_ci([5.0])["level"] is None


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
                            {"valid_runs": 0, "median": None})
    assert row["verdict"] == "unmeasurable"
    assert diag.overall_verdict([row])[0] == "INCONCLUSIVE"


def test_a_lane_without_an_interval_is_inconclusive_not_a_pass():
    """The decision needs the distribution-free intervals, not just the medians."""
    row = diag.compare_lane("essay", {"median": 100.0, "valid_runs": 21},
                            {"median": 101.0, "valid_runs": 21})
    assert row["verdict"] == "inconclusive"
    assert row["ratio_ci_low"] is None


def test_a_differing_shape_cannot_pass_on_the_shift_estimand():
    """The reviewer's counterexample, which a Hodges-Lehmann shift got wrong.

    Ten observations near 90 and eleven near 100 against eleven near 96 and ten
    near 110. The median ratio is 96/100 = 0.96, below structured's 0.97 band, so
    the lane is not non-inferior. A Hodges-Lehmann shift reported the median of
    the pairwise differences -- 100.0 - 90.0 = +10, comfortably above the
    boundary of -3.0 -- and so returned non-inferior while the ratio the band is
    actually written in sat below the band. The composed ratio interval puts its
    lower bound at 0.96, so it cannot certify the lane.
    """
    control = [90.0] * 10 + [100.0] * 11
    candidate = [96.0] * 11 + [110.0] * 10
    row = decide("structured", control, candidate)
    assert row["ratio"] == pytest.approx(0.96, abs=1e-9)
    assert row["band"] == 0.97
    assert row["ratio_ci_low"] < row["band"], row
    assert row["verdict"] != "non-inferior", row
    assert row["verdict"] == "inconclusive", row
    # And the shift-style estimand that produced the false pass is not consulted.
    assert "shift" not in row


def test_a_sample_below_the_coverage_requirement_is_inconclusive():
    """1, 2 and 3 observations cannot support the claimed 95% interval.

    An earlier version reported a pass at the claimed level for all three; the
    level it actually achieved was None, 0.0 and 0.5.
    """
    for n in (1, 2, 3, 5, 6):
        control = steady(100.0, n=n)
        candidate = steady(103.0, n=n)
        row = decide("essay", control, candidate)
        assert row["ratio"] > 0.95, n
        assert row["verdict"] == "inconclusive", (n, row)
        if n == 1:
            # A single observation supports no interval at all, which is itself
            # reported as None rather than as a level.
            assert row["ratio_ci_level"] is None
        else:
            assert row["ratio_ci_level"] < diag.MIN_LEVEL, (n, row)


def test_the_minimum_supported_sample_is_decided():
    """At exactly the minimum sample the lane is decided, not deferred."""
    n = diag.min_runs_for_level()
    assert decide("essay", steady(100.0, n=n), steady(103.0, n=n))["verdict"] == "non-inferior"


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
            "worker_container_started_at": f"{boot}-worker",
            "measure_container_started_at": boot,
            "measure_worker_container_started_at": f"{boot}-worker",
            "measure_verified_image": window.ARMS[arm]["tag"],
            "measure_verified_worker_image": window.ARMS[arm]["tag"],
            "measure_verified_exllamav3": window.ARMS[arm]["exllamav3"],
            "worker_exllamav3_version": window.ARMS[arm]["exllamav3"],
            "preemptions_delta": 0,
            "measure_attempt": f"{arm}-attempt-1",
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
                "attempt": state["arms"][arm]["measure_attempt"],
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
    assert "median ratio" in written["method"]["decision"].lower()
    assert "Hodges-Lehmann" in written["method"]["why_not_a_shift_interval"]


def test_the_running_phase_is_not_mistaken_for_a_crashed_one(tmp_path, monkeypatch):
    """`main` sets `phase_in_progress` BEFORE calling the handler.

    So a receipt read from inside the report phase legitimately has it set to
    "report". Treating that as a crash made every report run through `main()`
    return INVALID CAPTURE -- found by exercising the real bytes on the cluster,
    because a direct `phase_report(state)` call never sets the field.

    The in-run case is handled by `current_phase`, and a leftover for the
    terminal report phase is recoverable: re-running it regenerates the verdict
    from evidence that is still on disk. A leftover for any phase that produces
    evidence is flagged, whatever is running.
    """
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    state = _healthy_state(tmp_path)
    state["phase_in_progress"] = "report"
    # `--from report` requires a prior preflight, i.e. a recorded `.env` backup.
    state["backup"] = str(tmp_path / "env.bak")
    receipt.write_text(json.dumps(state))

    # The phase executing now is not evidence of an earlier crash.
    assert diag.validate_capture(state, current_phase="report") == []
    # Nor is a leftover report at any other moment: that phase only recomputes
    # the verdict, so a kill inside it leaves the capture intact.
    assert "report" not in diag.EVIDENCE_PHASES
    assert diag.validate_capture(state, current_phase=None) == []
    # A leftover for an evidence phase is flagged even while another runs.
    state["phase_in_progress"] = "measure_b"
    assert any("aborted inside phase 'measure_b'" in p
               for p in diag.validate_capture(state, current_phase="report"))

    # End to end through the real entry point: this must still pass.
    state["phase_in_progress"] = "report"
    receipt.write_text(json.dumps(state))
    assert diag.main(["--state", str(receipt), "--from", "report", "--to", "report"]) == 0
    written = json.loads(receipt.read_text())
    assert written["verdict"] == "NO REGRESSION DETECTED"
    assert written["capture_problems"] == []


@pytest.mark.parametrize("phase", ["rearm", "restore", "arm_b", "measure_a"])
def test_resuming_the_interrupted_evidence_phase_does_not_launder_it(
        tmp_path, monkeypatch, phase):
    """The reviewer's repro: `--from <the interrupted phase>` must not pass.

    `main` used to exempt a leftover equal to the phase being resumed, so a run
    killed inside `rearm` was accepted by simply resuming `rearm`: rc=0 and NO
    REGRESSION DETECTED, with the interruption erased. `restore` and the
    `arm_*`/`measure_*` phases had the same hole, and those phases leave no probe
    block behind, so nothing else preserved the failure.

    The handlers are stubbed: what is under test is the interruption
    bookkeeping, not the cluster work. `require_disarmed` is stubbed too, since
    an arm phase would otherwise reach for the cluster.
    """
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    monkeypatch.setattr(diag.window, "require_disarmed", lambda *a, **k: None)
    monkeypatch.setitem(diag.HANDLERS, phase, lambda state: None)
    state = _healthy_state(tmp_path)
    state["phase_in_progress"] = phase
    state["backup"] = str(tmp_path / "env.bak")
    receipt.write_text(json.dumps(state))

    # Resuming exactly the interrupted phase succeeds on its own terms...
    assert diag.main(["--state", str(receipt), "--from", phase, "--to", phase]) == 0
    written = json.loads(receipt.read_text())
    assert phase in written["interrupted_phases"]
    # ...but the interruption is not erasable, so judging the receipt fails
    # closed rather than reporting NO REGRESSION DETECTED.
    assert diag.main(["--state", str(receipt), "--from", "report", "--to", "report"]) == 1
    written = json.loads(receipt.read_text())
    assert written["verdict"] == "INVALID CAPTURE"
    assert any(f"interrupted inside phase '{phase}'" in p
               for p in written["capture_problems"])


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


def test_a_later_failure_is_not_erased_by_an_earlier_success(tmp_path, monkeypatch):
    """The reviewer's retry-history repro.

    An earlier attempt wrote a usable file and a later retry of the same lane
    failed. The old check kept only the latest entry per phase and accepted any
    successful block, so it read this lane as clean. Every registered attempt
    must have succeeded, because a capture that contained a failure is not a
    clean capture however the retry ended.
    """
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    state = _healthy_state(tmp_path)
    good = state["probes"]["a"]["essay"]
    state["probe_blocks"].append({
        "arm": "a", "lane": "essay", "runs": 21, "ok": False,
        "returncode": 1, "path": good, "error": "probe died on the retry",
        "attempt": state["arms"]["a"]["measure_attempt"],
    })

    problems = diag.validate_capture(state)
    assert any("did not succeed" in p for p in problems), problems
    diag.phase_report(state)
    assert json.loads(receipt.read_text())["verdict"] == "INVALID CAPTURE"


def test_a_failed_phase_is_not_erased_by_a_successful_retry(tmp_path, monkeypatch):
    """A phase that failed and then succeeded on retry must still be flagged."""
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    state = _healthy_state(tmp_path)
    state["phases"].append({"phase": "measure_b", "ok": False, "error": "died"})
    state["phases"].append({"phase": "measure_b", "ok": True})

    problems = diag.validate_capture(state)
    assert any("did not report success" in p for p in problems), problems


def test_a_phase_with_no_success_flag_is_not_evidence_of_success(tmp_path, monkeypatch):
    """Success must be AFFIRMATIVE.

    `ok` missing, or `ok: null`, is an incomplete contract: it says nothing
    about whether the phase succeeded. Testing only for `ok is False` accepted
    both, so a truncated or hand-edited receipt could pass a phase that never
    recorded its outcome.
    """
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    for entry in ({"phase": "measure_b"}, {"phase": "measure_b", "ok": None}):
        state = _healthy_state(tmp_path)
        state["phases"].append(entry)
        problems = diag.validate_capture(state)
        assert any("did not report success" in p for p in problems), (entry, problems)


def test_main_preserves_a_leftover_phase_instead_of_overwriting_it(tmp_path, monkeypatch):
    """End to end through the real entry point, on a receipt a killed run left.

    `main` sets `phase_in_progress` before each phase, so resuming with
    `--from report` overwrote the leftover `measure_a` and the interrupted retry
    read as a clean capture. It must be preserved and the run must fail closed.
    """
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    state = _healthy_state(tmp_path)
    state["phase_in_progress"] = "measure_a"
    state["backup"] = str(tmp_path / "env.bak")
    receipt.write_text(json.dumps(state))

    assert diag.main(["--state", str(receipt), "--from", "report", "--to", "report"]) == 1
    written = json.loads(receipt.read_text())
    assert written["verdict"] == "INVALID CAPTURE"
    assert "measure_a" in written["interrupted_phases"]
    assert any("interrupted inside phase 'measure_a'" in p
               for p in written["capture_problems"])


def test_an_interrupted_phase_recorded_by_main_blocks_the_verdict(tmp_path, monkeypatch):
    """`main` overwrites `phase_in_progress`; the prior value must survive."""
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    state = _healthy_state(tmp_path)
    state["interrupted_phases"] = ["measure_b"]

    problems = diag.validate_capture(state)
    assert any("interrupted inside phase 'measure_b'" in p for p in problems)
    diag.phase_report(state)
    assert json.loads(receipt.read_text())["verdict"] == "INVALID CAPTURE"


def test_evidence_from_an_earlier_attempt_is_not_accepted(tmp_path, monkeypatch):
    """The selected file must belong to the attempt being judged."""
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    state = _healthy_state(tmp_path)
    state["arms"]["a"]["measure_attempt"] = "a-attempt-2"

    problems = diag.validate_capture(state)
    assert any("does not belong to the current measurement attempt" in p
               for p in problems), problems


@pytest.mark.parametrize("strip", [
    lambda s: s.pop("counts"),
    lambda s: s.update(counts=None),
    lambda s: s.update(counts={}),
])
def test_a_missing_contract_survives_a_report_only_resume(tmp_path, monkeypatch, strip):
    """The reviewer's repro for the last hole in finding 3.

    `validate_capture` rejects an absent, null or empty `counts`, but `main`
    replaced all three with the CLI defaults *before* validation, so `--from
    report` on such a receipt returned rc=0 and NO REGRESSION DETECTED while
    writing a contract the capture never declared. A run that neither arms nor
    measures must keep the receipt's own record -- including its absence.
    """
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    state = _healthy_state(tmp_path)
    strip(state)
    # `--from report` needs a prior preflight, i.e. a recorded `.env` backup.
    state["backup"] = str(tmp_path / "env.bak")
    receipt.write_text(json.dumps(state))
    # Rejected when read directly...
    assert any("no usable observation count" in p
               for p in diag.validate_capture(state))

    # ...and the real entry point must not manufacture the missing contract.
    assert diag.main(["--state", str(receipt), "--from", "report", "--to", "report"]) == 1
    written = json.loads(receipt.read_text())
    assert written["verdict"] == "INVALID CAPTURE"
    assert any("no usable observation count" in p
               for p in written["capture_problems"])
    # The defaults must not have been written over the receipt's own record.
    assert not written["counts"]


def test_a_measuring_resume_of_a_contract_less_receipt_is_refused(
        tmp_path, monkeypatch, capsys):
    """A receipt holding measurements but no contract cannot be added to.

    Its evidence was collected under a contract nobody recorded, so defaulting one
    from these arguments would certify that evidence against a contract it never
    had. Refusing is the fail-closed answer.

    `backup` is set and the reason is pinned, so the exit status cannot come from
    some other precondition: without either, this test passes against a module
    that has no such guard at all.
    """
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    monkeypatch.setattr(diag.window, "require_disarmed", lambda *a, **k: None)
    # Stubbed so the test stays hermetic: if the refusal regresses, the run must
    # reach a verdict of 0 rather than reaching for the cluster.
    monkeypatch.setitem(diag.HANDLERS, "measure_b", lambda state: None)
    state = _healthy_state(tmp_path)
    state["backup"] = str(tmp_path / "env.bak")
    state.pop("counts")
    receipt.write_text(json.dumps(state))

    assert diag.main(["--state", str(receipt), "--from", "measure_b", "--to", "measure_b"]) == 2
    assert "no usable contract" in capsys.readouterr().err
    assert json.loads(receipt.read_text()).get("counts") is None


def test_an_arm_only_run_still_gets_a_contract(tmp_path, monkeypatch):
    """`consumes_contract` is wider than `measuring`: `phase_arm` reads counts.

    `--from arm_a --to arm_a` measures nothing, so its contract has to come from
    the arguments. Blanking it on the same rule would make `phase_arm` fail on
    `counts[lane]`, so the two conditions are kept distinct.
    """
    receipt = tmp_path / "diag.json"
    monkeypatch.setattr(diag.window, "_RECEIPT", receipt)
    monkeypatch.setattr(diag.window, "require_disarmed", lambda *a, **k: None)
    monkeypatch.setitem(diag.HANDLERS, "arm_a", lambda state: None)
    state = _healthy_state(tmp_path)
    # Nothing has measured yet, which is what `already_measured` asks.
    state["phases"] = [e for e in state["phases"]
                       if not e["phase"].startswith("measure_")]
    state["backup"] = str(tmp_path / "env.bak")
    state.pop("counts")
    receipt.write_text(json.dumps(state))

    assert diag.main(["--state", str(receipt), "--from", "arm_a", "--to", "arm_a"]) == 0
    assert json.loads(receipt.read_text())["counts"] == diag.DEFAULT_COUNTS


@pytest.mark.parametrize("mutate,expect", [
    (lambda s: s["phases"].append({"phase": "rearm", "ok": False, "error": "boom"}),
     "rearm"),
    (lambda s: s["phases"].pop(0), "preflight"),
    (lambda s: s.update(phase_in_progress="measure_b"), "measure_b"),
    (lambda s: s["arms"]["a"].update(preemptions_delta=None), "preemption delta unreadable"),
    (lambda s: s["arms"]["a"].update(measure_verified_exllamav3="1.4.9"), "exllamav3"),
    (lambda s: s["arms"]["b"].update(measure_verified_image="wrong:tag"), "not the arm's tag"),
    (lambda s: s["arms"]["a"].update(measure_container_started_at="boot-later"), "restarted"),
    # The reviewer's repro for finding 2: arm B's worker still on arm A's image.
    # The head was checked but the worker was not, so a two-node comparison whose
    # worker ran the other arm's build passed.
    (lambda s: s["arms"]["b"].update(
        measure_verified_worker_image=window.ARMS["a"]["tag"]), "worker"),
    (lambda s: s["arms"]["b"].update(worker_exllamav3_version="1.4.7"), "worker"),
    (lambda s: s["arms"]["a"].pop("worker_container_started_at"), "worker"),
    (lambda s: s["arms"]["a"].update(measure_worker_container_started_at="boot-later"),
     "worker"),
    # Same worker boot in both arms: the arms are not two distinct boots.
    (lambda s: s["arms"]["b"].update(
        worker_container_started_at="boot-a-worker",
        measure_worker_container_started_at="boot-a-worker"), "same worker"),
    # The reviewer's repro for finding 3: an absent count used to disable the
    # observation-length check entirely, so a short file passed a long contract.
    (lambda s: s["counts"].pop("hashmap"), "no usable observation count"),
    (lambda s: s["counts"].update(structured=None), "no usable observation count"),
    (lambda s: s["counts"].update(essay=0), "no usable observation count"),
    (lambda s: s["counts"].update(essay="21"), "no usable observation count"),
    # The receipt's count, the block's count and the file must agree.
    (lambda s: s["probe_blocks"][0].update(runs=7), "registered 7 runs"),
    (lambda s: s["arms"]["b"].update(container_started_at="boot-a"), "same head"),
    (lambda s: s["probes"]["a"].pop("hashmap"), "no evidence file was selected"),
    (lambda s: [b.update(ok=False, error="probe died")
                for b in s["probe_blocks"] if b["arm"] == "a" and b["lane"] == "essay"],
     "did not succeed"),
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
        ("short sample", "structured", {},
         f"expected {diag.DEFAULT_COUNTS['structured']}"),
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
