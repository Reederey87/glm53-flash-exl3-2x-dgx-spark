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

# Decode lanes are the ones the stall affects; they get the larger sample. The
# prefill lanes are kept at the §6 counts because a 240k prefill observation
# costs minutes, and prefill showed a tight settled distribution (0.0038 spread).
DECODE_LANES = ("structured", "essay", "hashmap")
PREFILL_LANES = ("prefill60k", "prefill240k")
DEFAULT_COUNTS = {
    "structured": 21,
    "essay": 21,
    "hashmap": 21,
    "prefill60k": 9,
    "prefill240k": 5,
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


def log(message: str) -> None:
    print(f"[{TAG}] {message}", flush=True)


# --- statistics -------------------------------------------------------------

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


def lane_stats(values: list[float], level: float = 0.95) -> dict:
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


def _mann_whitney_counts(m: int, n: int) -> list[int]:
    """Exact counts of the Mann-Whitney U statistic, as `counts[u]`.

    U is the number of pairs (a_i, b_j) with a_i < b_j. Building the merged
    ordering left to right: appending an `a` adds no such pair (it is last),
    while appending a `b` adds one for each `a` already placed. The recurrence
    is therefore `f[i][j][u] = f[i-1][j][u] + f[i][j-1][u-i]`, computed with
    integer arithmetic so the quantiles are exact.
    """
    f = [[[0] * (m * n + 1) for _ in range(n + 1)] for _ in range(m + 1)]
    f[0][0][0] = 1
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue
            for u in range(m * n + 1):
                total = 0
                if i:
                    total += f[i - 1][j][u]
                if j and u - i >= 0:
                    total += f[i][j - 1][u - i]
                f[i][j][u] = total
    return f[m][n]


def hodges_lehmann_ci(a: list[float], b: list[float], level: float = 0.95) -> dict:
    """Distribution-free CI for the shift `b - a`, plus its point estimate.

    This is the standard two-sample non-parametric comparison, and it is the one
    that actually answers the question the bands ask: whether `b` sits below `a`
    by more than the band allows. It is distribution-free (the Mann-Whitney
    statistic's null distribution depends only on the sample sizes), so like
    `median_ci` it assumes nothing about which observations are stalls.

    It is also strictly more informative than composing two separate median
    intervals, which needs a union bound over both and therefore discards the
    pairing between the two samples' spreads.
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return {"shift": None, "low": None, "high": None, "level": None,
                "n_pairs": 0, "order_statistic": None}
    diffs = sorted(bj - ai for bj in b for ai in a)
    shift = statistics.median(diffs)
    if m < 2 or n < 2:
        return {"shift": shift, "low": diffs[0], "high": diffs[-1], "level": None,
                "n_pairs": m * n, "order_statistic": None}
    counts = _mann_whitney_counts(m, n)
    total = sum(counts)
    alpha = (1.0 - level) / 2.0
    cumulative = 0
    c = 0
    for u in range(m * n + 1):
        cumulative += counts[u]
        if cumulative / total <= alpha:
            c = u
        else:
            break
    achieved = 1.0 - 2.0 * (sum(counts[:c + 1]) / total)
    return {
        "shift": shift,
        "low": diffs[c],
        "high": diffs[m * n - 1 - c],
        "level": achieved,
        "n_pairs": m * n,
        "order_statistic": c + 1,
    }


def run_values(kind: str, summary: dict) -> list[float]:
    key = "prefill_tok_s" if kind.startswith("prefill") else "tok_s"
    return [r[key] for r in summary.get("runs", []) if r.get(key) is not None]


def compare_lane(lane: str, control: dict, candidate: dict,
                 control_values: list[float] | None = None,
                 candidate_values: list[float] | None = None) -> dict:
    """Decide one lane from the two medians and a distribution-free shift CI.

    The point ratio answers the question; the shift interval decides whether the
    sample can answer it at all. The §6 band is a RATIO bound, so the interval is
    translated into ratio terms against the control's median: a shift of
    `(band - 1) * control_median` is exactly the band boundary. A lane whose
    interval spans that boundary is reported inconclusive rather than rounded to
    whichever side its point estimate fell on, which is what makes a wide or
    contaminated lane fail closed instead of quietly becoming a pass.
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
    if control_values is None or candidate_values is None:
        row.update({"verdict": "inconclusive",
                    "reason": "no raw observations available for the shift interval"})
        return row
    hl = hodges_lehmann_ci(control_values, candidate_values)
    row["shift"] = hl["shift"]
    row["shift_ci_low"] = hl["low"]
    row["shift_ci_high"] = hl["high"]
    row["shift_ci_level"] = hl["level"]
    # The band boundary expressed as a shift of the control's median.
    boundary = (band - 1.0) * a
    row["band_boundary_shift"] = boundary
    row["ratio_ci_low"] = (a + hl["low"]) / a
    row["ratio_ci_high"] = (a + hl["high"]) / a
    if hl["low"] >= boundary:
        row["verdict"] = "non-inferior"
        row["reason"] = (
            f"shift interval [{hl['low']:.4f}, {hl['high']:.4f}] lies at or above the "
            f"band boundary {boundary:.4f}"
        )
    elif hl["high"] < boundary:
        row["verdict"] = "REGRESSED"
        row["reason"] = (
            f"shift interval [{hl['low']:.4f}, {hl['high']:.4f}] lies below the "
            f"band boundary {boundary:.4f}"
        )
    else:
        row["verdict"] = "inconclusive"
        row["reason"] = (
            f"shift interval [{hl['low']:.4f}, {hl['high']:.4f}] spans the band "
            f"boundary {boundary:.4f}; the sample cannot decide this lane"
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
    phases = {p.get("phase"): p for p in state.get("phases", [])}

    for name in REQUIRED_PHASES:
        entry = phases.get(name)
        if entry is None:
            problems.append(f"phase {name} never ran")
        elif not entry.get("ok"):
            problems.append(f"phase {name} failed: {entry.get('error')}")
    for name, entry in phases.items():
        if entry.get("ok") is False and name not in REQUIRED_PHASES:
            problems.append(f"phase {name} failed: {entry.get('error')}")
    in_progress = state.get("phase_in_progress")
    if in_progress and in_progress != current_phase:
        problems.append(
            f"the run aborted inside phase {in_progress!r} "
            "(phase_in_progress was not cleared)"
        )

    counts = state.get("counts") or {}
    blocks = state.get("probe_blocks") or []
    for arm in ("a", "b"):
        record = (state.get("arms") or {}).get(arm) or {}
        expected = window.ARMS[arm]
        if record.get("measure_verified_image") != expected["tag"]:
            problems.append(
                f"arm {arm}: measured image {record.get('measure_verified_image')!r} "
                f"is not the arm's tag {expected['tag']!r}"
            )
        if record.get("measure_verified_exllamav3") != expected["exllamav3"]:
            problems.append(
                f"arm {arm}: measured exllamav3 {record.get('measure_verified_exllamav3')!r} "
                f"is not {expected['exllamav3']!r}"
            )
        # Boot binding: the arm phase's boot must be the one that was measured.
        boot = record.get("container_started_at")
        measured_boot = record.get("measure_container_started_at")
        if not boot or not measured_boot:
            problems.append(f"arm {arm}: missing container boot identity")
        elif boot != measured_boot:
            problems.append(
                f"arm {arm}: the container restarted between arming and measuring "
                f"({boot} -> {measured_boot})"
            )
        delta = record.get("preemptions_delta")
        if delta is None:
            problems.append(f"arm {arm}: preemption delta unreadable (not 'no preemptions')")
        elif delta != 0:
            problems.append(f"arm {arm}: {delta} preemptions during measurement")

    # The two arms must be different boots, or the comparison is not two arms.
    boots = {
        arm: ((state.get("arms") or {}).get(arm) or {}).get("container_started_at")
        for arm in ("a", "b")
    }
    if boots["a"] and boots["a"] == boots["b"]:
        problems.append("both arms report the same container boot identity")

    for arm in ("a", "b"):
        for lane in EXPECTED_LANES:
            want = counts.get(lane)
            matching = [b for b in blocks
                        if b.get("arm") == arm and b.get("lane") == lane]
            ok_blocks = [b for b in matching if b.get("ok") is True]
            if not matching:
                problems.append(f"arm {arm} lane {lane}: no probe block was registered")
            elif not ok_blocks:
                problems.append(
                    f"arm {arm} lane {lane}: every registered block failed "
                    f"({matching[-1].get('error')})"
                )
            selected = (state.get("probes") or {}).get(arm, {}).get(lane)
            if not selected:
                problems.append(f"arm {arm} lane {lane}: no evidence file was selected")
                continue
            if ok_blocks and ok_blocks[-1].get("path") != selected:
                problems.append(
                    f"arm {arm} lane {lane}: the selected file {selected!r} is not the "
                    f"one the successful block wrote ({ok_blocks[-1].get('path')!r})"
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
        values_by_arm: dict[str, list[float]] = {}
        for arm in ("a", "b"):
            summary = _load_lane(state, arm, lane)
            if summary is None:
                pair[arm] = {"valid_runs": 0, "median": None}
                values_by_arm[arm] = []
                continue
            values = run_values(lane, summary)
            values_by_arm[arm] = values
            entry = lane_stats(values)
            entry["invalid_runs"] = len(summary.get("invalid_runs", []))
            pair[arm] = entry
        stats[lane] = pair
        rows.append(compare_lane(lane, pair["a"], pair["b"],
                                 values_by_arm["a"], values_by_arm["b"]))
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
        "point_estimate": "median of each arm's observations",
        "decision": "Hodges-Lehmann shift (median of all pairwise B-A differences) "
                    "with its exact distribution-free Mann-Whitney confidence interval",
        "band_boundary": "(band - 1) x control median, i.e. the ratio band expressed "
                         "as a shift",
        "median_interval": "distribution-free sign-test order statistics (reported "
                           "per arm for transparency; not the decision rule)",
        "stall_count": "descriptive only; never used to certify the median it was "
                       "measured against",
        "fail_closed": "a lane whose shift interval spans its band boundary is "
                       "inconclusive; a receipt failing validate_capture() is "
                       "INVALID CAPTURE and cannot be read as a result",
    }
    window.save(state)

    print()
    print(f"{'lane':<12} {'A median':>10} {'B median':>10} {'B/A':>7} {'band':>5}  "
          f"{'shift CI (95%)':>24}  {'stalls A/B':>10}  verdict")
    for row in rows:
        lane = row["lane"]
        a, b = stats[lane]["a"], stats[lane]["b"]
        ratio = f"{row['ratio']:.4f}" if row.get("ratio") else "n/a"
        stalls = f"{a.get('stall_count', 0)}/{a.get('valid_runs', 0)} vs " \
                 f"{b.get('stall_count', 0)}/{b.get('valid_runs', 0)}"
        if row.get("shift_ci_low") is not None:
            interval = f"[{row['shift_ci_low']:+.2f}, {row['shift_ci_high']:+.2f}]"
        else:
            interval = "n/a"
        print(f"{lane:<12} {a.get('median') or float('nan'):>10.2f} "
              f"{b.get('median') or float('nan'):>10.2f} {ratio:>7} "
              f"{row['band']:>5.2f}  {interval:>24}  {stalls:>10}  {row['verdict']}")
    print()
    print("shift = B - A in the lane's own units; a positive shift means v1.4.9 is faster.")
    print("The band boundary is a shift of (band - 1) x A median, shown per lane below.")
    for row in rows:
        if row.get("band_boundary_shift") is not None:
            print(f"  {row['lane']:<12} boundary {row['band_boundary_shift']:+.4f}  "
                  f"point shift {row['shift']:+.4f}  -> {row['verdict']}")
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
    ap.add_argument("--decode-runs", type=int, default=DEFAULT_COUNTS["structured"])
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
        "structured": args.decode_runs,
        "essay": args.decode_runs,
        "hashmap": args.decode_runs,
        "prefill60k": args.prefill60k_runs,
        "prefill240k": args.prefill240k_runs,
    }
    if any(n < 1 for n in counts.values()):
        print("every lane needs at least one run", file=sys.stderr)
        return 2
    receipt = args.state or (ROOT / "local" / f"{TAG}-{time.strftime('%Y%m%d-%H%M%S')}.json")
    state: dict = {}
    if receipt.is_file():
        try:
            state = json.loads(receipt.read_text())
        except json.JSONDecodeError:
            state = {}
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
