#!/usr/bin/env python3
"""Offline judge for the task 35 §6 qualification window.

Reads the window receipt written by `run_v149_qualification_window.py` plus the
per-arm probe receipts it names, applies the contract that was pre-registered
before the window ran, and returns one verdict. It is deliberately separable
from the runner so the same capture can be re-judged after a parser fix, without
touching the cluster (the pattern used by the decode/prefill/occupancy oracles).

The question this answers is **non-inferiority**, not "did we find a win". Task
35 deployed the ExLlamaV3 v1.4.9 pin on an inertness analysis and a kernel
parity gate, and makes no performance claim; what was missing is the §6
observation contract. So the gate is: the candidate must not be slower than the
control by more than a pre-declared band, on a window whose own drift is inside
its band. A win is reported if it appears, but it is not required.

Verdicts
--------
ADOPT         every lane non-inferior, drift inside band, all correctness gates green
REVERT        a lane regressed beyond its band, or a correctness gate failed
INCONCLUSIVE  the window drifted or the evidence cannot support a decision
ABORT         the capture itself is unusable (missing arm, too few valid runs, NaN)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

# --- pre-registered contract (fixed before the window ran) -------------------
ARMS = ("a", "b", "b2", "a2")
CONTROL_ARMS = ("a", "a2")
CANDIDATE_ARMS = ("b", "b2")
LANES = ("structured", "essay", "hashmap", "prefill60k", "prefill240k")
REQUIRED_RUNS = {
    "structured": 9,
    "essay": 9,
    "hashmap": 9,
    "prefill60k": 5,
    "prefill240k": 5,
}
# Non-inferiority floor: candidate median must be >= this x control median.
BANDS = {
    "structured": 0.97,
    "essay": 0.95,
    "hashmap": 0.95,
    "prefill60k": 0.95,
    "prefill240k": 0.95,
}
# A window whose two control arms disagree by more than this cannot decide.
DRIFT_MAX = 0.05
# Arm identity: what the boot was supposed to be running.
ARM_IMAGE = {
    "a": "glm53-selfbuild:e3-w3-zfill",
    "b": "glm53-selfbuild:e3-w3-zfill-v149",
    "b2": "glm53-selfbuild:e3-w3-zfill-v149",
    "a2": "glm53-selfbuild:e3-w3-zfill",
}
ARM_EXLLAMAV3 = {"a": "1.4.7", "b": "1.4.9", "b2": "1.4.9", "a2": "1.4.7"}


def median(values: list[float]) -> float:
    return statistics.median(values)


def read_probe(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing probe receipt {path}")
    return json.loads(path.read_text())


def lane_values(probe: dict, lane: str) -> list[float]:
    key = "tok_s" if lane in ("structured", "essay", "hashmap") else "prefill_tok_s"
    values = [
        float(run[key])
        for run in probe.get("runs", [])
        if run.get(key) is not None and math.isfinite(float(run[key]))
    ]
    return values


def judge(receipt: dict, base_dir: Path) -> dict:
    errors: list[str] = []
    result: dict = {
        "schema": 1,
        "window": receipt.get("window"),
        "started": receipt.get("started"),
        "finished": receipt.get("finished"),
        "contract": {
            "arms": list(ARMS),
            "control_arms": list(CONTROL_ARMS),
            "candidate_arms": list(CANDIDATE_ARMS),
            "lanes": list(LANES),
            "required_runs": REQUIRED_RUNS,
            "bands": BANDS,
            "drift_max": DRIFT_MAX,
        },
        "errors": errors,
        "lanes": {},
    }

    # --- arm identity -------------------------------------------------------
    arms = receipt.get("arms") or {}
    for arm in ARMS:
        record = arms.get(arm)
        if not record:
            errors.append(f"arm {arm} missing from the window receipt")
            continue
        if record.get("image_tag") != ARM_IMAGE[arm]:
            errors.append(
                f"arm {arm} booted {record.get('image_tag')!r}, expected {ARM_IMAGE[arm]!r}"
            )
        if record.get("exllamav3_version") != ARM_EXLLAMAV3[arm]:
            errors.append(
                f"arm {arm} reports exllamav3 {record.get('exllamav3_version')!r}, "
                f"expected {ARM_EXLLAMAV3[arm]!r}"
            )
        if record.get("worker_image_tag") != ARM_IMAGE[arm]:
            errors.append(
                f"arm {arm} worker booted {record.get('worker_image_tag')!r}, "
                f"expected {ARM_IMAGE[arm]!r}"
            )
        if record.get("worker_exllamav3_version") != ARM_EXLLAMAV3[arm]:
            errors.append(
                f"arm {arm} worker reports exllamav3 "
                f"{record.get('worker_exllamav3_version')!r}, expected {ARM_EXLLAMAV3[arm]!r}"
            )
    if errors:
        result["verdict"] = "ABORT"
        return result

    # --- KV pool and preemption identity ------------------------------------
    # Every arm must reserve the same pool. Checking this per arm, rather than
    # only before-and-after, localizes a divergence to the boot that caused it.
    capacity_before = receipt.get("pool_capacity_before") or ""
    if not capacity_before:
        errors.append("pre-window KV pool capacity missing from the receipt")
    for arm in ARMS:
        record = arms.get(arm) or {}
        capacity = record.get("pool_capacity")
        if not capacity:
            errors.append(f"arm {arm} recorded no KV pool capacity")
        elif capacity_before and capacity != capacity_before:
            errors.append(
                f"arm {arm} KV pool capacity differs from the pre-window pool: "
                f"{capacity!r} vs {capacity_before!r}"
            )
        # A preemption during an arm means the measurement was disturbed.
        delta = record.get("preemptions_delta")
        if delta is None:
            errors.append(f"arm {arm} recorded no preemption delta")
        elif delta != 0:
            errors.append(f"arm {arm} saw {delta} preemptions during measurement")
    if errors:
        result["verdict"] = "ABORT"
        return result

    # --- correctness and safety gates ---------------------------------------
    gates = receipt.get("gates") or {}
    if gates.get("acceptance_rc") != 0:
        errors.append(f"acceptance battery rc={gates.get('acceptance_rc')!r} after restore")
    for node in ("memfree_head_gib", "memfree_worker_gib"):
        value = gates.get(node)
        # `NaN < 2.5` is False, so a plain comparison would let a NaN memory
        # reading through as "not below the tripwire". Require a finite number.
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"{node}={value!r} is not a finite memory reading")
        elif value < 2.5:
            errors.append(f"{node}={value!r} is below the 2.5 GiB tripwire")
    # Compare the PARSED capacity, not the raw line: the line carries a
    # timestamp, PID and source prefix that differ every boot.
    capacity_before = gates.get("pool_capacity_before") or ""
    capacity_after = gates.get("pool_capacity") or ""
    if not capacity_before:
        errors.append("pre-window KV pool capacity was not recorded")
    if not capacity_after:
        errors.append("KV pool capacity missing after restore")
    elif capacity_before and capacity_after != capacity_before:
        errors.append(
            f"KV pool capacity changed across the window: {capacity_before!r} -> {capacity_after!r}"
        )
    # The restored stamp is compared against the PRE-WINDOW stamp, not arm B's.
    # `prod-start.sh` hashes every raw `IMAGE=` line including overridden ones,
    # so arm B's `.env` (original + appended A + appended B) and the restored
    # `.env` (original only) hash differently even though both run B.
    stamp_before = gates.get("jit_stamp_before") or ""
    stamp_after = gates.get("jit_stamp") or ""
    if not stamp_before:
        errors.append("pre-window JIT shape stamp was not recorded")
    if not stamp_after:
        errors.append("JIT shape stamp missing after restore")
    elif stamp_before and stamp_after != stamp_before:
        errors.append(
            f"restored JIT shape stamp differs from the pre-window stamp "
            f"({stamp_before!r} -> {stamp_after!r})"
        )

    # --- per-lane evidence --------------------------------------------------
    probes = receipt.get("probes") or {}
    lane_medians: dict[str, dict[str, float]] = {}
    for lane in LANES:
        per_arm: dict[str, float] = {}
        for arm in ARMS:
            path = probes.get(arm, {}).get(lane)
            if not path:
                errors.append(f"probe receipt for arm {arm} lane {lane} was not recorded")
                continue
            probe_path = Path(path)
            if not probe_path.is_absolute():
                probe_path = base_dir / probe_path
            try:
                probe = read_probe(probe_path)
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"arm {arm} lane {lane}: {exc}")
                continue
            if probe.get("kind") != lane:
                errors.append(f"arm {arm} lane {lane}: receipt says kind {probe.get('kind')!r}")
                continue
            if probe.get("any_nan"):
                errors.append(f"arm {arm} lane {lane}: NaN/locklock marker in a decode run")
                continue
            if probe.get("any_cache_hit"):
                errors.append(f"arm {arm} lane {lane}: a cold-prefill run hit the prefix cache")
                continue
            # An EXCLUDED run that was excluded for a correctness reason is
            # still evidence: a NaN-corrupted output must not be silently
            # dropped into the exclusion list and then ignored.
            for bad in probe.get("invalid_runs") or []:
                reason = str(bad.get("invalid_reason") or "")
                if "NaN" in reason or "locklock" in reason:
                    errors.append(
                        f"arm {arm} lane {lane}: an excluded run was corrupted "
                        f"({reason})"
                    )
            valid = int(probe.get("valid_runs") or 0)
            if valid < REQUIRED_RUNS[lane]:
                errors.append(
                    f"arm {arm} lane {lane}: {valid} valid runs, "
                    f"§6 requires {REQUIRED_RUNS[lane]}"
                )
                continue
            values = lane_values(probe, lane)
            if len(values) < REQUIRED_RUNS[lane]:
                errors.append(f"arm {arm} lane {lane}: only {len(values)} usable values")
                continue
            per_arm[arm] = median(values)
        if len(per_arm) == len(ARMS):
            lane_medians[lane] = per_arm

    if errors:
        result["verdict"] = "ABORT"
        result["lane_medians"] = lane_medians
        return result

    # --- drift, ratio, verdict ---------------------------------------------
    worst = "ADOPT"
    for lane in LANES:
        per_arm = lane_medians[lane]
        control = median([per_arm[arm] for arm in CONTROL_ARMS])
        candidate = median([per_arm[arm] for arm in CANDIDATE_ARMS])
        a_first, a_last = per_arm["a"], per_arm["a2"]
        drift = abs(a_first - a_last) / max(a_first, a_last)
        b_drift = abs(per_arm["b"] - per_arm["b2"]) / max(per_arm["b"], per_arm["b2"])
        ratio = candidate / control
        # §6: report INCONCLUSIVE when variance prevents a decision. Candidate
        # instability is as disqualifying as control drift — a lane whose two
        # candidate arms disagree wildly has no settled number to compare, so
        # gating only the control would let a 75%-drift candidate pass as ADOPT.
        if drift > DRIFT_MAX or b_drift > DRIFT_MAX:
            lane_verdict = "INCONCLUSIVE"
        elif ratio < BANDS[lane]:
            lane_verdict = "FAIL"
        else:
            lane_verdict = "PASS"
        result["lanes"][lane] = {
            "per_arm_median": per_arm,
            "control_median": control,
            "candidate_median": candidate,
            "candidate_over_control": round(ratio, 4),
            "band": BANDS[lane],
            "control_drift": round(drift, 4),
            "candidate_drift": round(b_drift, 4),
            "verdict": lane_verdict,
        }
        if lane_verdict == "FAIL":
            worst = "REVERT"
        elif lane_verdict == "INCONCLUSIVE" and worst != "REVERT":
            worst = "INCONCLUSIVE"

    if worst == "REVERT":
        errors.append("a lane regressed beyond its pre-registered band")
    elif worst == "INCONCLUSIVE":
        errors.append(
            "the window drifted beyond its band (control or candidate); "
            "the comparison cannot decide"
        )
    result["verdict"] = worst
    # Be explicit about what an ADOPT does and does not certify. This harness
    # measures throughput and the correctness gates it collects; it does NOT
    # cover every §6 evidence requirement, so ADOPT is not a statement that the
    # full §6 qualification is complete.
    result["scope"] = {
        "covers": [
            "A-B-B-A ordering with the pre-registered per-lane observation counts",
            "decode (structured/essay/hashmap) and cold-prefill (60k/240k) throughput",
            "arm identity on both nodes (image tag + in-container exllamav3 version)",
            "KV pool capacity unchanged, per arm and across the window",
            "memory tripwire, preemption count, JIT shape stamp, NaN/cache-hit validity",
            "post-restore acceptance battery rc",
        ],
        "does_not_cover": [
            "temp-1 production cells (this harness runs temp-0 cells only)",
            "the §6 serving / toolcall / thinking-SSE / long-form / mixed-cache soak gates",
            "the prescribed drained-APC reset and cache-counter traffic audit for cold rounds",
        ],
        "adopt_means": (
            "no throughput regression beyond the pre-registered bands on the "
            "lanes measured; NOT that the full docs/13 §6 qualification is complete"
        ),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receipt", required=True, type=Path, help="window receipt JSON")
    ap.add_argument("--out", type=Path, help="write the audit JSON here (default: stdout)")
    args = ap.parse_args(argv)
    receipt = json.loads(args.receipt.read_text())
    result = judge(receipt, args.receipt.parent)
    text = json.dumps(result, indent=1, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, flush=True)
    verdict = result["verdict"]
    print(f"[audit] verdict: {verdict}", flush=True)
    for lane, row in result.get("lanes", {}).items():
        if "candidate_over_control" in row:
            print(
                f"[audit]   {lane:<12} {row['verdict']:<12} "
                f"ratio={row['candidate_over_control']:.4f} band={row['band']:.2f} "
                f"drift={row['control_drift']:.4f}",
                flush=True,
            )
    for message in result.get("errors", []):
        print(f"[audit]   ! {message}", flush=True)
    return 0 if verdict == "ADOPT" else 1


if __name__ == "__main__":
    sys.exit(main())
