#!/usr/bin/env python3
"""Paired two-arm diagnostic: is ExLlamaV3 v1.4.9 slower than v1.4.7?

NOT §6 QUALIFICATION EVIDENCE. This produces a diagnostic receipt only. It does
not satisfy the pre-registered §6 contract, it is not written by
`run_v149_qualification_window.py`, and `audit_v149_qualification.py` does not
consume it. §6 remains open until the registered window runs to completion or
the owner records a narrowing.

Why this exists
---------------
`run_v149_qualification_window.py` implements the pre-registered contract: 9
decode observations per arm per lane, and a within-arm variability gate of
`(max - min) / median <= 0.30`. That gate is maximally sensitive to a single
outlier. The head node produces a transient decode stall roughly once in twenty
observations (measured 2026-09-11: ~2.7x, position varies), so the gate fires,
the auditor returns INCONCLUSIVE, and the window spends about two hours and five
boots without deciding anything.

This asks the same question the §6 window asks -- "is v1.4.9 slower than v1.4.7
on these lanes?" -- with an estimator a couple of stalls cannot move. It takes
more observations per lane and compares MEDIANS, and it reports each median with
a **distribution-free confidence interval** from the sign test's order
statistics: the interval depends only on the sample size and the observed order
statistics, so it assumes nothing about the distribution and nothing about which
observations are stalls. A lane whose ratio interval spans its band is reported
inconclusive rather than rounded to whichever side its point estimate fell on.
Each lane's registered `(max - min) / median` spread is also reported, so the
receipt shows what the registered gate would have concluded.

An earlier version of this script gated on a `median_robust` flag that counted
observations below half their own median and called the lane trustworthy when
fewer than half qualified. That was circular and could not detect majority
contamination, and the median's 50% breakdown point does not mean replacing a
minority cannot shift it. It has been removed; `stall_count` survives as a
descriptive number that is never used to certify the median it was measured
against. See `median_ci()` for the replacement and `tests/test_v149_diagnostic.py`
for the regression tests.

The report phase validates the receipt's internal consistency BEFORE issuing a
verdict, because `--from report` and a resume after a late failure are both
reachable with every lane file present but the capture rejected. A receipt that
fails those checks is reported as INVALID CAPTURE and exits non-zero; the
per-lane numbers are still written as evidence, but they cannot be read as a
result.

Arms (same tags as the §6 window; one independent variable, the `IMAGE=` tag):

  A = glm53-selfbuild:e3-w3-zfill       (ExLlamaV3 1.4.7, control)
  B = glm53-selfbuild:e3-w3-zfill-v149  (ExLlamaV3 1.4.9, candidate)

A is measured first, then B, each with its own boot, because the revision is
baked into the process. Production is v1.4.9, so the restore phase puts the
pre-diagnostic `.env` back and reboots it.

Reuse
-----
Everything except the measure loop comes from the reviewed window runner:
`.env` handling, both-node arm verification, the MemFree tripwire, the
preemption check, the disarm/restore/re-arm recovery path, and the probe itself.
`phase_measure` is deliberately NOT reused, because its sample sizes ARE the
pre-registered §6 contract -- the §6 harness is left byte-identical and this
script implements its own loop.

Phases run in order and can be bounded with `--from/--to`:

  preflight  hashes, health, drain, both-node arm-image presence, `.env` backup
  disarm     stop the watchdog + metrics-alert timers
  arm_a      append arm A's `IMAGE=`, start via `local/prod-start.sh`, verify
  measure_a  run the diagnostic blocks on arm A
  arm_b      as `arm_a`, for the candidate
  measure_b  run the diagnostic blocks on arm B
  restore    put the pre-diagnostic `.env` back, reboot, verify v1.4.9 is up
  rearm      re-enable the timers
  report     compare the two arms on medians and print the verdict

Any failure after disarm reboots production from the pre-diagnostic `.env` and
re-arms the timers before exiting non-zero. SIGTERM/SIGHUP/SIGINT take the same
recovery path. The script never edits any `.env` line other than appending
`IMAGE=`. Resume with `--state <receipt> --from <phase>`.
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_v149_qualification as audit  # noqa: E402  (one source of truth for bands)
import run_v149_qualification_window as window  # noqa: E402  (reviewed primitives)

ROOT = window.ROOT
ENV_FILE = window.ENV_FILE
PROBE = window.PROBE
TAG = "task35b-diag"
EVIDENCE_CLASS = "diagnostic — NOT §6 qualification evidence"

# Decode lanes are the ones the stall affects, so they get the larger sample.
# The counts are set per lane from the pilot's own measured spread, not padded:
# `hashmap` has the widest genuine spread on this host (clean values span 26-33
# tok/s around a 29.5 median), so a 5% band on its median needs roughly four
# times the pilot's sample, while `structured` and `essay` clear their bands at
# 31 with power ~1.0. `prefill240k` is the opposite case -- its distribution is
# so tight (0.006 relative spread) that the interval decides even at its
# coverage minimum, but each observation costs ~152 s, so it stays near that
# minimum rather than matching the decode lanes.
DECODE_LANES = ("structured", "essay", "hashmap")
PREFILL_LANES = ("prefill60k", "prefill240k")
DEFAULT_COUNTS = {
    "structured": 31,
    "essay": 31,
    "hashmap": 81,
    "prefill60k": 31,
    # 11 rather than the 7-observation coverage minimum: at 7 the order
    # statistic is k=0, so the interval includes both extremes and one stall
    # would break the lane. 11 buys k=1.
    "prefill240k": 11,
}
LANE_TIMEOUTS = {
    "structured": 3600.0,
    "essay": 3600.0,
    "hashmap": 3600.0,
    "prefill60k": 3600.0,
    "prefill240k": 5400.0,
}
# A stall is a run at less than half the lane's own median. The observed stalls
# are ~2.7x, so half is a conservative detector: it cannot mistake ordinary
# spread for a stall.
STALL_FRACTION = 0.5
PHASES = (
    "preflight",
    "disarm",
    "arm_a",
    "measure_a",
    "arm_b",
    "measure_b",
    "restore",
    "rearm",
    "report",
)
ARM_PHASES = tuple(name for name in PHASES if name.startswith(("arm_", "measure_")))
# Phases that only recompute their output from inputs that are already on disk.
# Re-running one reproduces exactly what the killed run would have produced, so
# an interruption there leaves no gap in the evidence. Every other phase either
# touches the cluster or records state, and an interrupted one means the capture
# may be incomplete -- that interruption must never be discarded.
RECOMPUTABLE_PHASES = ("report",)
EVIDENCE_PHASES = tuple(name for name in PHASES if name not in RECOMPUTABLE_PHASES)


def log(message: str) -> None:
    print(f"[{TAG}] {message}", flush=True)


# --- statistics -------------------------------------------------------------

# The decision is about the RATIO of the two arms' medians, so the uncertainty
# has to be an interval for that ratio. Each arm's median interval is built at
# ARM_LEVEL so that the composition reaches MIN_LEVEL by the union bound:
# coverage >= 1 - (1 - ARM_LEVEL) - (1 - ARM_LEVEL) = 2 * ARM_LEVEL - 1.
MIN_LEVEL = 0.95
ARM_LEVEL = 1.0 - (1.0 - MIN_LEVEL) / 2.0

# An earlier version decided with a Hodges-Lehmann shift interval. That estimates
# the median of pairwise `B - A` DIFFERENCES, which equals the difference of
# medians only under a location-shift model, so across differently shaped arms it
# does not bound the median ratio at all: with A = ten near 90 and eleven near
# 100, and B = eleven near 96 and ten near 110, the median ratio is 0.9610 --
# below structured's 0.97 band -- while the shift interval reported [1.060,
# 1.100] and called it non-inferior. The composed median intervals below target
# the ratio itself and assume nothing about either arm's shape.


def _binomial_cdf(k: int, n: int) -> float:
    """P(Bin(n, 1/2) <= k), exactly, as a fraction of 2**n.

    Used only to size the distribution-free interval below. Integer arithmetic
    throughout: the probabilities are exact dyadic rationals, so there is no
    floating-point drift in the choice of order statistic.
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = 1 << n
    return sum(math.comb(n, i) for i in range(k + 1)) / total


def median_ci(values: list[float], level: float = 0.95) -> dict:
    """Distribution-free confidence interval for the median, from order statistics.

    This is the interval justified WITHOUT reference to the data's own spread, so
    it cannot be circular. The sign test gives
    `P(x_(k+1) <= median <= x_(n-k)) = 1 - 2*P(Bin(n, 1/2) <= k)`, so picking the
    largest `k` whose binomial tail is inside `(1 - level)/2` yields an exact
    interval that assumes nothing about the distribution and nothing about which
    observations are outliers.

    It replaces an earlier `median_robust` flag that was CIRCULAR: it called a
    lane robust when fewer than half the observations fell below half the median,
    but for positive values and odd `n` fewer than half always do, so the flag
    could not detect majority contamination (11 observations at 30 with 10 at 100
    gave median 30 and reported "robust"). The median's 50% breakdown point is a
    statement about how far it can be moved, not a licence to skip measuring how
    far it was moved, and it does not mean replacing a minority cannot shift it.

    When `n` is too small for the requested level, the widest available interval
    is returned together with the level it actually achieves, rather than
    pretending to a precision the sample does not support.
    """
    clean = sorted(float(v) for v in values)
    n = len(clean)
    if n == 0:
        return {"low": None, "high": None, "level": None, "n": 0}
    if n == 1:
        return {"low": clean[0], "high": clean[0], "level": None, "n": 1}
    alpha = (1.0 - level) / 2.0
    k = 0
    for candidate in range(n):
        if _binomial_cdf(candidate, n) <= alpha:
            k = candidate
        else:
            break
    # k is the largest tail index inside alpha; the interval is the (k+1)-th
    # smallest to the (k+1)-th largest.
    achieved = 1.0 - 2.0 * _binomial_cdf(k, n)
    return {
        "low": clean[k],
        "high": clean[n - 1 - k],
        "level": achieved,
        "n": n,
        "order_statistic": k + 1,
    }


def min_runs_for_level(level: float = MIN_LEVEL) -> int:
    """Smallest sample size whose composed median-ratio interval reaches `level`.

    The composed interval's coverage depends only on `n` and the order-statistic
    rule, never on the observed values, so the requirement can be checked before
    a run starts rather than discovered afterwards. With `level=0.95` this is 7:
    a 5- or 6-observation arm cannot support it.
    """
    arm_level = 1.0 - (1.0 - level) / 2.0
    for n in range(2, 128):
        ci = median_ci([float(i) for i in range(n)], level=arm_level)
        if ci["level"] is not None and 2.0 * ci["level"] - 1.0 >= level:
            return n
    return 128


def lane_stats(values: list[float], level: float = ARM_LEVEL) -> dict:
    """Median, its distribution-free interval, and descriptive spread measures.

    `stall_count` is DESCRIPTIVE. It is deliberately not a trustworthiness gate:
    it is measured against the lane's own median, so it cannot be used to certify
    that same median. What bounds the median here is `median_ci`, whose width
    comes from the sample size and the observed order statistics alone.
    """
    clean = sorted(float(v) for v in values)
    n = len(clean)
    if not n:
        return {"valid_runs": 0, "median": None, **median_ci([])}
    median = statistics.median(clean)
    if not median:
        return {"valid_runs": n, "median": None, **median_ci(clean)}
    mad = statistics.median([abs(v - median) for v in clean])
    ci = median_ci(clean, level=level)
    stalls = [v for v in clean if v < STALL_FRACTION * median]
    return {
        "valid_runs": n,
        "median": median,
        "min": clean[0],
        "max": clean[-1],
        "mad": mad,
        "mad_over_median": mad / median,
        "stall_threshold": STALL_FRACTION * median,
        "stall_count": len(stalls),
        "stall_fraction": len(stalls) / n,
        # What the registered §6 variability gate would have computed.
        "registered_spread": (clean[-1] - clean[0]) / median,
        "registered_settled": (clean[-1] - clean[0]) / median <= audit.VARIABILITY_MAX,
        **ci,
    }


def run_values(kind: str, summary: dict) -> list[float]:
    key = "prefill_tok_s" if kind.startswith("prefill") else "tok_s"
    return [r[key] for r in summary.get("runs", []) if r.get(key) is not None]


def compare_lane(lane: str, control: dict, candidate: dict) -> dict:
    """Decide one lane from a distribution-free interval for the MEDIAN RATIO.

    Each arm's `median_ci` is a valid interval for that arm's own median at
    `ARM_LEVEL`, so pairing the candidate's low bound with the control's high
    bound (and vice versa) gives a valid, conservative interval for the ratio
    `B_median / A_median` at `2 * ARM_LEVEL - 1` by the union bound. That is the
    quantity the §6 bands are written in, so no shape or location-shift
    assumption is needed and no different estimand is silently substituted.

    Two ways a lane fails closed: its coverage cannot reach `MIN_LEVEL` at this
    sample size, or its interval spans the band. Either yields `inconclusive`
    rather than being rounded to whichever side the point estimate fell on.
    """
    band = audit.BANDS[lane]
    a, b = control.get("median"), candidate.get("median")
    row: dict = {"lane": lane, "band": band, "control_median": a, "candidate_median": b}
    if not a or not b:
        row.update({"ratio": None, "verdict": "unmeasurable",
                    "reason": "a lane produced no valid observation"})
        return row
    ratio = b / a
    row["ratio"] = ratio
    row["candidate_over_control_percent"] = (ratio - 1.0) * 100.0

    levels = [control.get("level"), candidate.get("level")]
    bounds = [control.get("low"), control.get("high"),
              candidate.get("low"), candidate.get("high")]
    if any(v is None for v in levels + bounds):
        row.update({"ratio_ci_low": None, "ratio_ci_high": None, "ratio_ci_level": None,
                    "verdict": "inconclusive",
                    "reason": "no distribution-free median interval is available for "
                              "this lane"})
        return row
    composed = 2.0 * min(levels) - 1.0
    # The ratio is smallest when the candidate is at its low bound and the
    # control at its high bound, and largest the other way round.
    lo = candidate["low"] / control["high"]
    hi = candidate["high"] / control["low"]
    row.update({"ratio_ci_low": lo, "ratio_ci_high": hi, "ratio_ci_level": composed})

    if composed < MIN_LEVEL:
        row["verdict"] = "inconclusive"
        row["reason"] = (
            f"this sample supports only {composed:.4f} coverage for the median ratio "
            f"({control.get('valid_runs')} vs {candidate.get('valid_runs')} observations); "
            f"the requested {MIN_LEVEL:.2f} cannot be achieved, so the lane is not decided"
        )
    elif lo >= band:
        row["verdict"] = "non-inferior"
        row["reason"] = (
            f"median-ratio interval [{lo:.4f}, {hi:.4f}] at {composed:.4f} coverage lies "
            f"at or above band {band:.2f}"
        )
    elif hi < band:
        row["verdict"] = "REGRESSED"
        row["reason"] = (
            f"median-ratio interval [{lo:.4f}, {hi:.4f}] at {composed:.4f} coverage lies "
            f"below band {band:.2f}"
        )
    else:
        row["verdict"] = "inconclusive"
        row["reason"] = (
            f"median-ratio interval [{lo:.4f}, {hi:.4f}] spans band {band:.2f}; the "
            "sample cannot decide this lane"
        )
    return row


def overall_verdict(rows: list[dict]) -> tuple[str, str]:
    if any(r["verdict"] == "REGRESSED" for r in rows):
        return "REGRESSION", "a lane's median fell below its pre-registered band"
    if any(r["verdict"] in ("inconclusive", "unmeasurable") for r in rows):
        return "INCONCLUSIVE", "at least one lane could not be decided"
    return "NO REGRESSION DETECTED", (
        "every lane's shift interval lies at or above its pre-registered band boundary"
    )


# --- phases -----------------------------------------------------------------

def phase_preflight(state: dict) -> None:
    for path in (ENV_FILE, PROBE):
        if not path.is_file():
            raise RuntimeError(f"missing {path}")
    backup = ROOT / f".env.bak-pre-{TAG}-{time.strftime('%Y%m%d-%H%M%S')}"
    window.shutil.copy2(ENV_FILE, backup)
    state.update({"backup": str(backup), "env_sha256": window.win.sha256(ENV_FILE)})
    window.save(state)
    env = window.effective_env_all()
    if env.get("IMAGE") != window.PRODUCTION_IMAGE:
        raise RuntimeError(
            f"preflight expects production on {window.PRODUCTION_IMAGE!r}, "
            f".env says {env.get('IMAGE')!r}"
        )
    code, _ = window.win.curl("/health", timeout=10)
    if code != 200:
        raise RuntimeError(f"production is not healthy before the diagnostic (/health -> {code})")
    if not window.win.drain():
        raise RuntimeError("server did not drain; refusing to take it down")
    # Both arms must already exist on BOTH nodes: shipping one mid-diagnostic
    # would add a variable (and a multi-gigabyte transfer) to the comparison.
    presence: dict[str, dict] = {}
    for arm in ("a", "b"):
        tag = window.ARMS[arm]["tag"]
        head_ok = window.win.run(["docker", "image", "inspect", tag], timeout=60, check=False).returncode == 0
        worker_ok = window.win.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", window.WORKER,
             f"docker image inspect {tag}"], timeout=90, check=False,
        ).returncode == 0
        presence[arm] = {"tag": tag, "head": head_ok, "worker": worker_ok}
        if not (head_ok and worker_ok):
            raise RuntimeError(f"arm {arm} image {tag!r} missing (head={head_ok} worker={worker_ok})")
    state.update({
        "evidence_class": EVIDENCE_CLASS,
        "runner_sha256": window.win.sha256(Path(__file__)),
        "probe_sha256": window.win.sha256(PROBE),
        "prod_start_sha256": window.win.sha256(ROOT / "local" / "prod-start.sh"),
        "start_sha256": window.win.sha256(ROOT / "start.sh"),
        "env_effective_before": env,
        "image_before": window.win.image_id(),
        "jit_stamp_before": window.jit_stamp(),
        "pool_capacity_before": window.pool_capacity(),
        "arm_image_presence": presence,
        "contract": {"arms": {k: window.ARMS[k] for k in ("a", "b")}, "lanes": dict(state["counts"])},
    })
    log(f"preflight OK backup={backup.name} pool={state['pool_capacity_before'][:60]}")


def phase_arm(state: dict, arm: str) -> None:
    """Boot one arm. Delegates to the reviewed window runner's arm phase."""
    window.phase_arm(state, arm)


def phase_measure(state: dict, arm: str) -> None:
    """Run the diagnostic blocks on the running arm.

    Mirrors the reviewed `phase_measure`'s safety properties (re-verify the arm,
    bind the observations to the verified boot, tripwire each lane, register the
    block before running it, fail on a non-zero probe, refuse a raised
    preemption count) with the diagnostic's own sample sizes.
    """
    counts = state["counts"]
    head_now = window.verify_arm(arm, window.HEAD_CONTAINER)
    worker_now = window.verify_arm(arm, window.WORKER_CONTAINER, host=window.WORKER)
    record = state.setdefault("arms", {}).setdefault(arm, {})
    boot_now = window.container_started_at()
    worker_boot_now = window.container_started_at(window.WORKER_CONTAINER, host=window.WORKER)
    # A missing boot identity is a refusal, not a pass: an unavailable token
    # would silently bypass the binding the check exists to enforce.
    if not boot_now or not worker_boot_now:
        raise RuntimeError(
            f"arm {arm}: could not read the container boot identity "
            f"(head={boot_now!r} worker={worker_boot_now!r}); refusing to measure"
        )
    recorded_boot = record.get("container_started_at")
    recorded_worker_boot = record.get("worker_container_started_at")
    if not recorded_boot or not recorded_worker_boot:
        raise RuntimeError(
            f"arm {arm}: no boot identity was recorded by the arm phase; resume "
            "from the arm phase so the observations belong to a verified boot"
        )
    if boot_now != recorded_boot or worker_boot_now != recorded_worker_boot:
        raise RuntimeError(
            f"arm {arm}: a container restarted since the arm phase "
            f"(head {recorded_boot} -> {boot_now}, worker {recorded_worker_boot} -> "
            f"{worker_boot_now}); the observations would not belong to the verified boot"
        )
    record.update({
        "measure_verified_image": head_now["image_tag"],
        "measure_verified_worker_image": worker_now["image_tag"],
        "measure_verified_exllamav3": head_now["exllamav3_version"],
        "measure_container_started_at": boot_now,
        "measure_worker_container_started_at": worker_boot_now,
    })
    attempts = int(state.get("measure_attempts", 0)) + 1
    state["measure_attempts"] = attempts
    attempt = f"{time.strftime('%Y%m%d-%H%M%S')}-{attempts}"
    record["measure_attempt"] = attempt
    window.save(state)

    state.setdefault("probes", {}).setdefault(arm, {})
    for lane in (*DECODE_LANES, *PREFILL_LANES):
        runs = int(counts[lane])
        out = window._RECEIPT.parent / f"{window._RECEIPT.stem}-{arm}-{lane}-{attempt}.json"
        head_before, worker_before = window.check_tripwire(f"arm {arm} {lane} before")
        # Register the attempt BEFORE running it, so a failed block cannot
        # vanish from the receipt and a later judge would see the failure.
        block = {
            "arm": arm, "lane": lane, "runs": runs, "attempt": attempt,
            "path": out.name, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "ok": None, "returncode": None, "error": None,
            "memfree_head_gib": [head_before, None],
            "memfree_worker_gib": [worker_before, None],
        }
        state.setdefault("probe_blocks", []).append(block)
        window.save(state)
        started = time.time()
        proc = subprocess.run(
            ["python3", str(PROBE), "--kind", lane, "--runs", str(runs), "--out", str(out)],
            check=False, text=True, timeout=LANE_TIMEOUTS[lane],
        )
        elapsed = time.time() - started
        head_after, worker_after = window.check_tripwire(f"arm {arm} {lane} after")
        block.update({
            "seconds": round(elapsed, 1),
            "memfree_head_gib": [head_before, head_after],
            "memfree_worker_gib": [worker_before, worker_after],
            "returncode": proc.returncode,
        })
        if proc.returncode != 0:
            tail = (proc.stdout or "") + (proc.stderr or "")
            block["ok"] = False
            block["error"] = tail[-400:] or "(no output)"
            block["evidence_present"] = out.is_file()
            window.save(state)
            raise RuntimeError(
                f"arm {arm} lane {lane}: probe exited {proc.returncode}; "
                f"last output: {tail[-400:] or '(no output)'}"
            )
        block["ok"] = True
        state["probes"][arm][lane] = out.name
        window.save(state)
        log(f"arm {arm} lane {lane}: {runs} runs in {elapsed:.0f}s")
    before = record.get("preemptions_before")
    after = window.preemptions()
    record["preemptions_after"] = after
    # None on either side is "could not read", not "no preemptions"; propagate it
    # rather than treating an unread counter as zero.
    delta = None if before is None or after is None else after - before
    record["preemptions_delta"] = delta
    window.save(state)
    if delta is None:
        raise RuntimeError(
            f"arm {arm}: preemption telemetry unavailable (before={before!r} "
            f"after={after!r}); refusing to treat an unread counter as no preemptions"
        )
    if delta:
        raise RuntimeError(f"arm {arm}: preemptions increased by {delta} during measurement")


def phase_restore(state: dict) -> None:
    window.phase_restore(state)


def phase_rearm(state: dict) -> None:
    window.phase_rearm(state)


def _evidence_dir() -> Path | None:
    """Directory holding the evidence files, or None when no receipt is set.

    The window runner owns `_RECEIPT`; the report and its validation read the
    lane files relative to it. Resolved through this helper so the dependency is
    explicit and a caller without a receipt gets a clear failure rather than an
    `AttributeError` on None.
    """
    return window._RECEIPT.parent if window._RECEIPT is not None else None


def _load_lane(state: dict, arm: str, lane: str) -> dict | None:
    name = state.get("probes", {}).get(arm, {}).get(lane)
    if not name:
        return None
    base = _evidence_dir()
    if base is None:
        return None
    path = base / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


# The phases that must have completed for the numbers to mean anything. `report`
# is excluded because it is the phase being run; `judge` is not a diagnostic
# phase.
REQUIRED_PHASES = ("preflight", "disarm", "arm_a", "measure_a", "arm_b",
                   "measure_b", "restore", "rearm")
EXPECTED_LANES = (*DECODE_LANES, *PREFILL_LANES)


def validate_capture(state: dict, current_phase: str | None = None) -> list[str]:
    """Every reason this receipt must NOT be read as a passing verdict.

    The report phase is reachable by `--from report`, and by a resume after a
    failure late in the run — in both cases the lane files can all be present
    while the capture itself was rejected. Reading those files and printing
    NO REGRESSION DETECTED would launder a failed run into a pass, so the
    receipt is checked for internal consistency BEFORE any verdict is issued.

    `current_phase` is the phase executing right now. `main` records the phase
    in progress *before* calling its handler and clears it afterwards, so a
    receipt read from inside a phase legitimately has `phase_in_progress` set to
    that phase; only a DIFFERENT value means an earlier run died mid-phase.

    Checks are deliberately independent: each returns its own message, and one
    failure does not stop the others from being reported.
    """
    problems: list[str] = []
    entries = state.get("phases", [])
    seen = {e.get("phase") for e in entries}

    for name in REQUIRED_PHASES:
        if name not in seen:
            problems.append(f"phase {name} never ran")
    # Iterate every entry, not the latest per phase: a retry that succeeded must
    # not erase the record of the attempt that failed before it. Success must be
    # AFFIRMATIVE: an entry with no `ok` field, or `ok: null`, is not evidence
    # that the phase succeeded, and testing only for `False` let both through.
    for entry in entries:
        if entry.get("ok") is not True:
            problems.append(
                f"phase {entry.get('phase')} did not report success "
                f"(ok={entry.get('ok')!r}): {entry.get('error')}"
            )
    in_progress = state.get("phase_in_progress")
    if in_progress and in_progress != current_phase and in_progress in EVIDENCE_PHASES:
        problems.append(
            f"the run aborted inside phase {in_progress!r} "
            "(phase_in_progress was not cleared)"
        )
    # `main` records the phase it is about to run, so a leftover value from an
    # earlier run is overwritten and lost. It preserves them here instead.
    for leftover in state.get("interrupted_phases") or []:
        problems.append(
            f"an earlier attempt was interrupted inside phase {leftover!r}; "
            "the capture is incomplete"
        )

    counts = state.get("counts") or {}
    blocks = state.get("probe_blocks") or []
    # The counts are the contract the evidence was collected under, so a missing
    # or nonsensical one must be a failure rather than a skipped check: an absent
    # count used to disable the observation-length test entirely, letting a
    # 7-observation file stand in for a 31-observation contract.
    for lane in EXPECTED_LANES:
        want = counts.get(lane)
        if not isinstance(want, int) or isinstance(want, bool) or want < 1:
            problems.append(
                f"lane {lane}: the receipt records no usable observation count "
                f"({want!r}); the evidence cannot be checked against a contract"
            )
    for arm in ("a", "b"):
        record = (state.get("arms") or {}).get(arm) or {}
        expected = window.ARMS[arm]
        # Both nodes are checked, not just the head. The comparison is a
        # two-node measurement, so a worker running a different image would
        # invalidate it just as surely as a wrong head. The worker's exllamav3
        # version is the one the arm phase recorded rather than a measure-time
        # re-read (the runner re-verifies the worker's boot at measure time but
        # not its version), so it is the strongest worker evidence available.
        for node, image_key, version_key in (
            ("head", "measure_verified_image", "measure_verified_exllamav3"),
            ("worker", "measure_verified_worker_image", "worker_exllamav3_version"),
        ):
            if record.get(image_key) != expected["tag"]:
                problems.append(
                    f"arm {arm} {node}: measured image {record.get(image_key)!r} "
                    f"is not the arm's tag {expected['tag']!r}"
                )
            if record.get(version_key) != expected["exllamav3"]:
                problems.append(
                    f"arm {arm} {node}: measured exllamav3 {record.get(version_key)!r} "
                    f"is not {expected['exllamav3']!r}"
                )
        # Boot binding, per node: the boot the arm phase recorded must be the one
        # that was measured, or the observations belong to some other boot.
        for node, armed_key, measured_key in (
            ("head", "container_started_at", "measure_container_started_at"),
            ("worker", "worker_container_started_at", "measure_worker_container_started_at"),
        ):
            armed = record.get(armed_key)
            measured = record.get(measured_key)
            if not armed or not measured:
                problems.append(f"arm {arm} {node}: missing container boot identity")
            elif armed != measured:
                problems.append(
                    f"arm {arm} {node}: the container restarted between arming and "
                    f"measuring ({armed} -> {measured})"
                )
        delta = record.get("preemptions_delta")
        if delta is None:
            problems.append(f"arm {arm}: preemption delta unreadable (not 'no preemptions')")
        elif delta != 0:
            problems.append(f"arm {arm}: {delta} preemptions during measurement")

    # The two arms must be different boots, or the comparison is not two arms.
    for node, key in (("head", "container_started_at"),
                      ("worker", "worker_container_started_at")):
        boots = {
            arm: ((state.get("arms") or {}).get(arm) or {}).get(key)
            for arm in ("a", "b")
        }
        if boots["a"] and boots["a"] == boots["b"]:
            problems.append(
                f"both arms report the same {node} container boot identity"
            )

    for arm in ("a", "b"):
        attempt = ((state.get("arms") or {}).get(arm) or {}).get("measure_attempt")
        for lane in EXPECTED_LANES:
            want = counts.get(lane)
            matching = [b for b in blocks
                        if b.get("arm") == arm and b.get("lane") == lane]
            # EVERY registered attempt must have succeeded. A retry that
            # eventually worked still means the capture contained a failure, and
            # accepting the later success would hide it -- the same reason the §6
            # judge rejects any receipt carrying a failed block.
            failed = [b for b in matching if b.get("ok") is not True]
            if not matching:
                problems.append(f"arm {arm} lane {lane}: no probe block was registered")
            elif failed:
                problems.append(
                    f"arm {arm} lane {lane}: {len(failed)} of {len(matching)} registered "
                    f"attempt(s) did not succeed (last: {failed[-1].get('error')!r})"
                )
            selected = (state.get("probes") or {}).get(arm, {}).get(lane)
            if not selected:
                problems.append(f"arm {arm} lane {lane}: no evidence file was selected")
                continue
            # The block that wrote the selected file must itself claim the
            # contract's observation count, so the three records -- the receipt's
            # count, the block's count, and the file's observations -- have to
            # agree rather than each being checked in isolation.
            chosen_blocks = [b for b in matching if b.get("path") == selected]
            if chosen_blocks and isinstance(want, int) and chosen_blocks[-1].get("runs") != want:
                problems.append(
                    f"arm {arm} lane {lane}: the block that wrote the selected file "
                    f"registered {chosen_blocks[-1].get('runs')!r} runs, but the "
                    f"receipt's count is {want}"
                )
            # Bind the selected evidence to the CURRENT measurement attempt, so a
            # stale file from an earlier attempt cannot be judged as this one's.
            if not attempt:
                problems.append(f"arm {arm} lane {lane}: the arm recorded no measurement attempt")
            else:
                chosen = [b for b in matching
                          if b.get("attempt") == attempt and b.get("path") == selected]
                if not chosen:
                    problems.append(
                        f"arm {arm} lane {lane}: the selected file {selected!r} does not "
                        f"belong to the current measurement attempt {attempt!r}"
                    )
            summary = _load_lane(state, arm, lane)
            if summary is None:
                problems.append(f"arm {arm} lane {lane}: evidence file missing or unreadable")
                continue
            if summary.get("kind") != lane:
                problems.append(
                    f"arm {arm} lane {lane}: evidence file reports kind "
                    f"{summary.get('kind')!r}"
                )
            invalid = summary.get("invalid_runs") or []
            if invalid:
                problems.append(
                    f"arm {arm} lane {lane}: {len(invalid)} invalid run(s) in the evidence"
                )
            values = run_values(lane, summary)
            if want is not None and len(values) != want:
                problems.append(
                    f"arm {arm} lane {lane}: {len(values)} valid observation(s), expected {want}"
                )
            bad = [v for v in values if not math.isfinite(v) or v <= 0]
            if bad:
                problems.append(
                    f"arm {arm} lane {lane}: {len(bad)} non-finite or non-positive value(s)"
                )
            # The probe's own coldness flag. A prefill lane that hit the cache
            # did not measure what this lane claims to measure.
            if lane.startswith("prefill") and summary.get("any_cache_hit"):
                problems.append(f"arm {arm} lane {lane}: any_cache_hit is true")
    return problems


def phase_report(state: dict) -> None:
    lanes = list(EXPECTED_LANES)
    problems = validate_capture(state, current_phase="report")
    stats: dict[str, dict] = {}
    rows: list[dict] = []
    for lane in lanes:
        pair: dict[str, dict] = {}
        for arm in ("a", "b"):
            summary = _load_lane(state, arm, lane)
            if summary is None:
                pair[arm] = {"valid_runs": 0, "median": None, "level": None,
                             "low": None, "high": None}
                continue
            entry = lane_stats(run_values(lane, summary))
            entry["invalid_runs"] = len(summary.get("invalid_runs", []))
            pair[arm] = entry
        stats[lane] = pair
        rows.append(compare_lane(lane, pair["a"], pair["b"]))
    state["stats"] = stats
    state["comparison"] = rows
    state["capture_problems"] = problems

    if problems:
        # A rejected capture outranks every lane result. The per-lane numbers are
        # still written, because they are the evidence of what was captured, but
        # they must never be readable as a verdict.
        verdict = "INVALID CAPTURE"
        reason = (
            f"{len(problems)} integrity problem(s) in the receipt; the comparison "
            "below is NOT a verdict. First: " + problems[0]
        )
    else:
        verdict, reason = overall_verdict(rows)
    state["verdict"] = verdict
    state["verdict_reason"] = reason
    state["evidence_class"] = EVIDENCE_CLASS
    state["method"] = {
        "point_estimate": "median of each arm's observations; the ratio B_median/A_median",
        "decision": "conservative composed interval for the MEDIAN RATIO: each arm's "
                    "distribution-free sign-test order-statistic interval at "
                    f"{ARM_LEVEL:.4f}, paired across arms so the ratio is covered at "
                    f"{MIN_LEVEL:.2f} by the union bound",
        "why_not_a_shift_interval": "a Hodges-Lehmann shift estimates the median of "
                                    "pairwise differences, which equals the difference "
                                    "of medians only under a location-shift model; it "
                                    "does not bound the median ratio across differently "
                                    "shaped arms",
        "coverage_requirement": f"a lane below {MIN_LEVEL:.2f} coverage is inconclusive; "
                                f"the minimum sample supporting it is "
                                f"{min_runs_for_level()} observations per arm",
        "stall_count": "descriptive only; never used to certify the median it was "
                       "measured against",
        "fail_closed": "a lane whose ratio interval spans its band is inconclusive; a "
                       "receipt failing validate_capture() is INVALID CAPTURE and cannot "
                       "be read as a result",
    }
    window.save(state)

    print()
    print(f"{'lane':<12} {'A median':>10} {'B median':>10} {'B/A':>7} {'band':>5}  "
          f"{'median-ratio CI':>22}  {'cov':>6}  stalls A/B  verdict")
    for row in rows:
        lane = row["lane"]
        a, b = stats[lane]["a"], stats[lane]["b"]
        ratio = f"{row['ratio']:.4f}" if row.get("ratio") else "n/a"
        stalls = f"{a.get('stall_count', 0)}/{a.get('valid_runs', 0)} vs " \
                 f"{b.get('stall_count', 0)}/{b.get('valid_runs', 0)}"
        if row.get("ratio_ci_low") is not None:
            interval = f"[{row['ratio_ci_low']:.4f}, {row['ratio_ci_high']:.4f}]"
        else:
            interval = "n/a"
        cov = f"{row['ratio_ci_level']:.4f}" if row.get("ratio_ci_level") else "n/a"
        print(f"{lane:<12} {a.get('median') or float('nan'):>10.2f} "
              f"{b.get('median') or float('nan'):>10.2f} {ratio:>7} "
              f"{row['band']:>5.2f}  {interval:>22}  {cov:>6}  {stalls}  {row['verdict']}")
    print()
    print("The interval covers the MEDIAN RATIO B_median/A_median at the stated coverage,")
    print(f"composed from two {ARM_LEVEL:.4f} per-arm intervals by the union bound.")
    print(f"A lane below {MIN_LEVEL:.2f} coverage, or whose interval spans its band, is")
    print(f"inconclusive. The minimum sample that supports {MIN_LEVEL:.2f} is")
    print(f"{min_runs_for_level()} observations per arm.")
    print()
    if problems:
        print("CAPTURE PROBLEMS (the verdict below is not a result):")
        for problem in problems:
            print(f"  - {problem}")
        print()
    print(f"VERDICT: {verdict} — {reason}")
    print(f"evidence class: {EVIDENCE_CLASS}")
    settled = sum(1 for r in rows if stats[r["lane"]]["a"].get("registered_settled"))
    print(f"registered §6 variability gate would have been satisfied on {settled}/{len(rows)} arms")
    print()
    print()


HANDLERS = {
    "preflight": phase_preflight,
    "disarm": window.phase_disarm,
    "arm_a": lambda state: phase_arm(state, "a"),
    "measure_a": lambda state: phase_measure(state, "a"),
    "arm_b": lambda state: phase_arm(state, "b"),
    "measure_b": lambda state: phase_measure(state, "b"),
    "restore": phase_restore,
    "rearm": phase_rearm,
    "report": phase_report,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="first", choices=PHASES, default=PHASES[0])
    ap.add_argument("--to", dest="last", choices=PHASES, default=PHASES[-1])
    ap.add_argument("--state", type=Path, help="receipt/checkpoint file")
    ap.add_argument("--decode-runs", type=int,
                    help="set all three decode lanes to the same count")
    ap.add_argument("--structured-runs", type=int, default=DEFAULT_COUNTS["structured"])
    ap.add_argument("--essay-runs", type=int, default=DEFAULT_COUNTS["essay"])
    ap.add_argument("--hashmap-runs", type=int, default=DEFAULT_COUNTS["hashmap"])
    ap.add_argument("--prefill60k-runs", type=int, default=DEFAULT_COUNTS["prefill60k"])
    ap.add_argument("--prefill240k-runs", type=int, default=DEFAULT_COUNTS["prefill240k"])
    ap.add_argument("--keep-armed", action="store_true",
                    help="on failure leave the arm boot running (debug only)")
    args = ap.parse_args(argv)
    lo, hi = PHASES.index(args.first), PHASES.index(args.last)
    if lo > hi:
        print("--from must not come after --to", file=sys.stderr)
        return 2
    counts = {
        "structured": args.decode_runs or args.structured_runs,
        "essay": args.decode_runs or args.essay_runs,
        "hashmap": args.decode_runs or args.hashmap_runs,
        "prefill60k": args.prefill60k_runs,
        "prefill240k": args.prefill240k_runs,
    }
    if any(n < 1 for n in counts.values()):
        print("every lane needs at least one run", file=sys.stderr)
        return 2
    # Fail fast rather than spending the boots: a lane whose sample cannot reach
    # the required coverage can never be decided, so measuring it is wasted time.
    measuring = any(name.startswith("measure_") for name in PHASES[lo:hi + 1])
    minimum = min_runs_for_level()
    if measuring:
        too_small = {lane: n for lane, n in counts.items() if n < minimum}
        if too_small:
            print(
                f"these lane counts cannot support {MIN_LEVEL:.2f} coverage for the "
                f"median ratio (minimum {minimum} observations per arm): "
                + ", ".join(f"{lane}={n}" for lane, n in sorted(too_small.items())),
                file=sys.stderr,
            )
            return 2
    receipt = args.state or (ROOT / "local" / f"{TAG}-{time.strftime('%Y%m%d-%H%M%S')}.json")
    state: dict = {}
    if receipt.is_file():
        try:
            state = json.loads(receipt.read_text())
        except json.JSONDecodeError:
            state = {}
    # `counts` is the contract the evidence was collected under, and
    # `validate_capture` checks each lane file against it. Overwriting it from
    # the current arguments would rewrite history: re-judging a receipt taken
    # with 5 prefill240k observations against a default of 7 would report an
    # integrity failure that never happened. So preserve it when this invocation
    # measures nothing, and refuse to mix samples when it does.
    recorded = state.get("counts")
    if recorded and recorded != counts:
        already_measured = any(
            str(e.get("phase", "")).startswith("measure_") for e in state.get("phases", [])
        )
        if measuring and already_measured:
            print(
                f"{receipt} already measured under {recorded}; these arguments "
                f"({counts}) would mix samples from two contracts",
                file=sys.stderr,
            )
            return 2
        if not measuring:
            counts = recorded
    state.update({
        "schema": 1,
        "kind": TAG,
        "evidence_class": EVIDENCE_CLASS,
        "counts": counts,
        "started": state.get("started") or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    state.setdefault("phases", [])
    if args.first != PHASES[0] and "backup" not in state:
        print(f"--from {args.first} needs a prior preflight state (no backup in {receipt})", file=sys.stderr)
        return 2
    # Hand the shared module its receipt and recovery context, so `save`,
    # `emergency_restore` and the signal path all operate on THIS run.
    window._RECEIPT = receipt
    window._ACTIVE = state
    window._KEEP_ARMED = args.keep_armed
    window._RESTORE_DONE = False
    if args.first in ARM_PHASES:
        try:
            window.require_disarmed(f"--from {args.first}")
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, window._on_signal)
    atexit.register(window.emergency_restore)
    # A leftover `phase_in_progress` means an earlier attempt died inside that
    # phase without clearing it -- `main` clears the field after every phase,
    # success or failure, and saves, so a persisted value can only come from a
    # hard kill (SIGKILL, OOM, power loss, host stall). The loop below overwrites
    # the field before every phase, so the signal would otherwise be lost and a
    # later success would read as a clean capture. Preserve it instead.
    #
    # This covers the phase this invocation is about to resume, not just the
    # others. An earlier revision exempted the match, reasoning that `main` sets
    # the field before calling the handler and so a resuming run looks identical
    # to a starting one. That reasoning was wrong: the in-run case is already
    # handled by `validate_capture`'s `current_phase`, while exempting the match
    # discarded real evidence -- a run killed during `restore` or `arm_*` was
    # silently accepted by resuming that same phase, and those phases leave no
    # probe block to preserve the failure.
    #
    # The one exception is a phase that only recomputes its output, i.e. the
    # terminal report: re-running it regenerates the verdict from evidence that
    # is still on disk, so a kill there leaves the capture itself intact.
    # Flagging it would permanently poison a receipt for a millisecond-wide
    # window, which on this cluster (host stalls, power capping) is a real
    # possibility. `EVIDENCE_PHASES` is every phase that does NOT qualify.
    leftover = state.get("phase_in_progress")
    if leftover and leftover in EVIDENCE_PHASES:
        state.setdefault("interrupted_phases", []).append(leftover)
        log(f"an earlier attempt was interrupted inside phase {leftover!r}")
    state["phase_in_progress"] = None
    window.save(state)

    failure: BaseException | None = None
    try:
        for name in PHASES[lo:hi + 1]:
            state["phase_in_progress"] = name
            window.save(state)
            log(f"--- phase {name} ---")
            try:
                HANDLERS[name](state)
            except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
                state["phases"].append({"phase": name, "ok": False, "error": repr(exc)})
                state["phase_in_progress"] = None
                failure = exc
                log(f"phase {name} FAILED: {exc}")
                window.save(state)
                break
            state["phases"].append({"phase": name, "ok": True})
            state["phase_in_progress"] = None
            window.save(state)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
        failure = failure or exc
        log(f"diagnostic aborted outside a phase handler: {exc!r}")
    finally:
        if failure is not None and not args.keep_armed:
            window.restore_production(state)
            window.recover_timers(state)
            window._RESTORE_DONE = window._recovery_ok(state)
        else:
            window._RESTORE_DONE = True
        state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            window.save(state)
        except OSError as exc:
            log(f"could not write the final receipt: {exc!r}")

    log(f"receipt: {receipt}")
    if failure is not None:
        return 1
    # A run that completed without a passing verdict is not a success. Without
    # this, `--from report` on a damaged receipt would exit 0 while printing
    # INVALID CAPTURE, and a caller checking only the exit status would read the
    # failure as a pass.
    if state.get("verdict") and state["verdict"] != "NO REGRESSION DETECTED":
        log(f"verdict {state['verdict']!r} is not a pass; exiting 1")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
