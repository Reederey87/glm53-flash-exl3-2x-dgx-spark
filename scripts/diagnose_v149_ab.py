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
more observations per lane and compares MEDIANS. A median cannot be moved by
fewer than half the observations, so it is a valid central estimate while fewer
than half the runs are stalls; that property is reported per lane rather than
assumed. Each lane's registered `(max - min) / median` spread is also reported,
so the receipt shows what the registered gate would have concluded.

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

def lane_stats(values: list[float]) -> dict:
    """Median and a stall-aware robustness statement for one lane.

    `median_robust` is a property of the estimator, not a chosen tolerance: the
    median of n values is unchanged by any subset of fewer than ceil(n/2)
    observations, so a stall count below half the sample cannot move it.
    """
    clean = sorted(float(v) for v in values)
    n = len(clean)
    if not n:
        return {"valid_runs": 0, "median": None, "median_robust": False}
    median = statistics.median(clean)
    if not median:
        return {"valid_runs": n, "median": None, "median_robust": False}
    mad = statistics.median([abs(v - median) for v in clean])
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
        "median_robust": len(stalls) * 2 < n,
        # What the registered §6 variability gate would have computed.
        "registered_spread": (clean[-1] - clean[0]) / median,
        "registered_settled": (clean[-1] - clean[0]) / median <= audit.VARIABILITY_MAX,
    }


def run_values(kind: str, summary: dict) -> list[float]:
    key = "prefill_tok_s" if kind.startswith("prefill") else "tok_s"
    return [r[key] for r in summary.get("runs", []) if r.get(key) is not None]


def compare_lane(lane: str, control: dict, candidate: dict) -> dict:
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
    if not (control.get("median_robust") and candidate.get("median_robust")):
        row["verdict"] = "inconclusive"
        row["reason"] = (
            f"too many stalls to trust the median "
            f"(control {control.get('stall_count')}/{control.get('valid_runs')}, "
            f"candidate {candidate.get('stall_count')}/{candidate.get('valid_runs')})"
        )
    elif ratio >= band:
        row["verdict"] = "non-inferior"
    else:
        row["verdict"] = "REGRESSED"
        row["reason"] = f"median ratio {ratio:.4f} below band {band:.2f}"
    return row


def overall_verdict(rows: list[dict]) -> tuple[str, str]:
    if any(r["verdict"] == "REGRESSED" for r in rows):
        return "REGRESSION", "a lane's median fell below its pre-registered band"
    if any(r["verdict"] in ("inconclusive", "unmeasurable") for r in rows):
        return "INCONCLUSIVE", "at least one lane could not be decided"
    return "NO REGRESSION DETECTED", (
        "every lane's median is at or above its pre-registered band, with the "
        "median trustworthy on both arms"
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


def _load_lane(state: dict, arm: str, lane: str) -> dict | None:
    name = state.get("probes", {}).get(arm, {}).get(lane)
    if not name:
        return None
    path = window._RECEIPT.parent / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def phase_report(state: dict) -> None:
    lanes = (*DECODE_LANES, *PREFILL_LANES)
    stats: dict[str, dict] = {}
    rows: list[dict] = []
    for lane in lanes:
        pair: dict[str, dict] = {}
        for arm in ("a", "b"):
            summary = _load_lane(state, arm, lane)
            if summary is None:
                pair[arm] = {"valid_runs": 0, "median": None, "median_robust": False}
                continue
            values = run_values(lane, summary)
            entry = lane_stats(values)
            entry["invalid_runs"] = len(summary.get("invalid_runs", []))
            pair[arm] = entry
        stats[lane] = pair
        rows.append(compare_lane(lane, pair["a"], pair["b"]))
    verdict, reason = overall_verdict(rows)
    state["stats"] = stats
    state["comparison"] = rows
    state["verdict"] = verdict
    state["verdict_reason"] = reason
    state["evidence_class"] = EVIDENCE_CLASS
    window.save(state)

    print()
    print(f"{'lane':<12} {'A median':>10} {'B median':>10} {'B/A':>7} {'band':>5}  "
          f"{'stalls A/B':>10}  {'spread A/B (registered)':>26}  verdict")
    for row in rows:
        lane = row["lane"]
        a, b = stats[lane]["a"], stats[lane]["b"]
        ratio = f"{row['ratio']:.4f}" if row.get("ratio") else "n/a"
        stalls = f"{a.get('stall_count', 0)}/{a.get('valid_runs', 0)} vs " \
                 f"{b.get('stall_count', 0)}/{b.get('valid_runs', 0)}"
        spread = f"{a.get('registered_spread', float('nan')):.3f} / " \
                 f"{b.get('registered_spread', float('nan')):.3f}"
        print(f"{lane:<12} {a.get('median') or float('nan'):>10.2f} "
              f"{b.get('median') or float('nan'):>10.2f} {ratio:>7} "
              f"{row['band']:>5.2f}  {stalls:>10}  {spread:>26}  {row['verdict']}")
    print()
    print(f"VERDICT: {verdict} — {reason}")
    print(f"evidence class: {EVIDENCE_CLASS}")
    settled = sum(1 for r in rows if stats[r["lane"]]["a"].get("registered_settled"))
    print(f"registered §6 variability gate would have been satisfied on {settled}/{len(rows)} arms")
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
    return 1 if failure is not None else 0


if __name__ == "__main__":
    sys.exit(main())
