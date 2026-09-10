#!/usr/bin/env python3
"""Task 24 W5 gate: judge an ncu occupancy capture of the E3 grouped kernels.

The W5 contract (docs/11 §8) opens an occupancy sweep only when ncu at
production shapes shows achieved occupancy well below the theoretical value,
and stops when the kernel already needs more than 96 registers or the gap is
small. This auditor reads the CSV that ``ncu --csv`` writes for
``fm_gateup_kernel`` / ``fm_down_kernel``, extracts the launch geometry and the
Occupancy section, and returns one verdict per kernel plus an overall decision.

It fails closed: a missing kernel, a missing required metric, a launch whose
metric values are outside their domain, an unreadable metric value (``N/A``,
``ERROR (...)``), a row without a launch id, or an achieved occupancy above the
theoretical one is an ABORT, not a silent pass. Records are kept per launch, so
one incomplete launch of a kernel cannot borrow another launch's numbers; the
per-kernel values used for the gate are the minimum across that kernel's
launches (the conservative direction: a lower achieved occupancy is likelier to
open the sweep, and a lower register count is likelier to allow it). It does no
CUDA work and never contacts a GPU, so it runs on the Mac against a captured
CSV.

Decisions:
  STOP_REGISTER_HEADROOM  registers/thread > --regs-stop; a higher-occupancy
                          target would spill, so the sweep must not open.
  STOP_NO_GAP             achieved >= --achieved-ratio x theoretical; the
                          kernel is at its structural occupancy ceiling.
  OPEN_GAP                achieved is far below theoretical and the register
                          stop does not apply; W5 may open with a candidate.
  ABORT                   capture incomplete, unusable or unparseable.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

# Kernel names ncu reports for the E3 grouped path (demangled base names).
KERNELS = ("fm_gateup_kernel", "fm_down_kernel")

# Canonical metric -> (csv labels, raw metric names)
METRICS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "registers_per_thread": (("Registers Per Thread",), ("launch__registers_per_thread",)),
    "block_size": (("Block Size",), ("launch__block_size",)),
    "grid_size": (("Grid Size",), ("launch__grid_size",)),
    "smem_static": (("Static Shared Memory Per Block",), ("launch__shared_mem_per_block_static",)),
    "smem_dynamic": (("Dynamic Shared Memory Per Block",), ("launch__shared_mem_per_block_dynamic",)),
    "block_limit_registers": (("Block Limit Registers",), ("launch__occupancy_limit_registers",)),
    "block_limit_shared_mem": (("Block Limit Shared Mem",), ("launch__occupancy_limit_shared_mem",)),
    "block_limit_warps": (("Block Limit Warps",), ("launch__occupancy_limit_warps",)),
    "block_limit_sm": (("Block Limit SM",), ("launch__occupancy_limit_blocks",)),
    "theoretical_occupancy_pct": (("Theoretical Occupancy",), ("sm__maximum_warps_per_active_cycle_pct",)),
    "achieved_occupancy_pct": (
        ("Achieved Occupancy",),
        ("sm__warps_active.avg.pct_of_peak_sustained_active",),
    ),
    "waves_per_sm": (("Waves Per SM",), ("launch__waves_per_multiprocessor",)),
}

REQUIRED = (
    "registers_per_thread",
    "block_size",
    "grid_size",
    "block_limit_registers",
    "block_limit_shared_mem",
    "block_limit_warps",
    "theoretical_occupancy_pct",
    "achieved_occupancy_pct",
    "waves_per_sm",
)

_NUMBER = re.compile(r"\s*([-+]?\d+(?:\.\d+)?)\s*([A-Za-z%][A-Za-z%/]*)?\s*")


def _number(text: str) -> float | None:
    """Parse an ncu metric value, or None when it is not a plain number.

    The whole value must be a number with an optional unit token
    ("128 register/thread", "33.33 %", "4608"). Anything else — ``N/A``,
    ``ERROR (33.02)``, an empty cell — is unparseable, and a metric we
    recognise but cannot read must fail the capture closed rather than let a
    stray substring stand in for a measurement.
    """
    match = _NUMBER.fullmatch(text or "")
    return float(match.group(1)) if match else None


def _lookup(metric: str) -> dict[str, str]:
    labels, names = METRICS[metric]
    table: dict[str, str] = {}
    for label in labels:
        table[label.strip().lower()] = metric
    for name in names:
        table[name.strip().lower()] = metric
    return table


def _csv_body(text: str) -> str:
    """Drop the ncu log preamble (warnings, ==PROF== lines) before the header."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "Metric Name" in line:
            return "\n".join(lines[index:])
    raise ValueError("capture has no ncu CSV header ('Metric Name')")


def parse_csv(text: str) -> dict[str, dict[str, dict[str, float | None]]]:
    """Return {kernel: {launch id: {canonical_metric: value}}} from an ncu capture.

    Launches stay separate: a kernel profiled twice (ncu `--launch-skip`/`--launch-count`
    can capture the same kernel more than once) must not have one launch's metrics
    silently stand in for another's. A metric we recognise but cannot read is kept
    with a ``None`` value so the launch fails closed instead of losing a metric.
    """
    lookup: dict[str, str] = {}
    for metric in METRICS:
        lookup.update(_lookup(metric))
    reader = csv.DictReader(io.StringIO(_csv_body(text)))
    if reader.fieldnames is None:
        raise ValueError("empty capture")
    fields = {name.strip().lower(): name for name in reader.fieldnames if name}
    for needed in ("metric name", "metric value"):
        if needed not in fields:
            raise ValueError(f"capture has no '{needed}' column: {reader.fieldnames}")
    kernel_key = None
    for candidate in ("kernel name", "function name", "kernel"):
        if candidate in fields:
            kernel_key = fields[candidate]
            break
    if kernel_key is None:
        raise ValueError(f"capture has no kernel-name column: {reader.fieldnames}")
    metric_key = fields["metric name"]
    value_key = fields["metric value"]
    id_key = fields.get("id")
    if id_key is None:
        # Without launch IDs two profiles of the same kernel would merge, and one
        # launch's numbers could stand in for another's. ncu always writes them.
        raise ValueError(f"capture has no launch-id column: {reader.fieldnames}")

    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for row in reader:
        kernel = (row.get(kernel_key) or "").strip()
        launch = (row.get(id_key) or "").strip()
        label = (row.get(metric_key) or "").strip().lower()
        canonical = lookup.get(label)
        if not kernel:
            # A row carrying a launch id and a metric we recognise is a real
            # measurement whose kernel identity is missing; dropping it would
            # silently remove a whole launch from the gate. Rows that are blank
            # or carry only unrelated columns are skipped as before.
            if launch and canonical is not None:
                raise ValueError(
                    f"capture row for {canonical!r} (launch {launch!r}) has no kernel name"
                )
            continue
        short = next((name for name in KERNELS if name in kernel), None)
        if short is None:
            continue
        if not launch:
            raise ValueError(f"capture row for {short} has no launch id")
        # The launch record is created from the first row seen for that launch,
        # whatever its metric and value look like, so a launch with unreadable
        # metrics is a missing-metric failure rather than an invisible one.
        record = out.setdefault(short, {}).setdefault(launch, {})
        if canonical is None:
            continue
        if record.get(canonical) is not None:
            # A readable value already stands for this metric; ncu can emit the
            # same metric under both its label and its raw name.
            continue
        record[canonical] = _number(row.get(value_key) or "")
    return out


def _launch_order(launch: str) -> tuple[int, str]:
    return (int(launch), "") if launch.isdigit() else (1 << 30, launch)


def _launch_problem(kernel: str, launch: str,
                    metrics: dict[str, float | None]) -> str | None:
    """Return why this launch record cannot be judged, or None when it is usable."""
    where = f"{kernel} launch {launch}"
    missing = [key for key in REQUIRED if key not in metrics]
    if missing:
        return f"{where}: missing metrics {missing}"
    unreadable = sorted(key for key, value in metrics.items() if value is None)
    if unreadable:
        # A metric the capture states but does not spell as a number ("N/A",
        # "ERROR (...)") is not a measurement; it must not be ignored.
        return f"{where}: unreadable metric value(s) {unreadable}"
    regs = metrics["registers_per_thread"]
    block = metrics["block_size"]
    grid = metrics["grid_size"]
    theoretical = metrics["theoretical_occupancy_pct"]
    achieved = metrics["achieved_occupancy_pct"]
    if regs <= 0:
        return f"{where}: registers/thread {regs:g} is not a usable value"
    if block <= 0:
        return f"{where}: block size {block:g} is not a usable value"
    if grid <= 0:
        return f"{where}: grid size {grid:g} is not a usable value"
    if not 0 < theoretical <= 100:
        return f"{where}: theoretical occupancy {theoretical:g}% is outside (0, 100]"
    if not 0 < achieved <= 100:
        return f"{where}: achieved occupancy {achieved:g}% is outside (0, 100]"
    if achieved > theoretical + 0.05:
        return (f"{where}: achieved occupancy {achieved:g}% exceeds theoretical "
                f"{theoretical:g}%")
    for key in ("block_limit_registers", "block_limit_shared_mem", "block_limit_warps",
                "waves_per_sm"):
        if metrics[key] < 0:
            return f"{where}: {key} {metrics[key]:g} is negative"
    return None


def judge(
    capture: dict[str, dict[str, dict[str, float | None]]],
    *,
    regs_stop: float = 96.0,
    achieved_ratio: float = 0.8,
    expected_launches: int | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "regs_stop": regs_stop,
        "achieved_ratio": achieved_ratio,
        "aggregation": "min across launches per kernel",
        "kernels": {},
        "decision": "ABORT",
        "reason": "",
    }
    missing_kernels = [name for name in KERNELS if name not in capture]
    if missing_kernels:
        report["reason"] = f"capture missing kernels: {missing_kernels}"
        return report
    total_launches = sum(len(launches) for launches in capture.values())
    if expected_launches is not None and total_launches != expected_launches:
        # ncu was asked for this many launches; a capture with fewer records means
        # a launch was lost (or malformed) and the per-launch gate would be weaker
        # than pre-registered.
        report["reason"] = (
            f"capture holds {total_launches} launch record(s), expected "
            f"{expected_launches}"
        )
        return report

    for name in KERNELS:
        launches = capture[name]
        if not launches:
            report["reason"] = f"{name}: capture has no launch record"
            return report
        keysets = {frozenset(record) for record in launches.values()}
        if len(keysets) != 1:
            report["reason"] = (
                f"{name}: its launches do not report the same metrics "
                f"{sorted(sorted(keys) for keys in keysets)}"
            )
            return report
        for launch in sorted(launches, key=_launch_order):
            problem = _launch_problem(name, launch, launches[launch])
            if problem:
                report["reason"] = problem
                return report

    verdicts: list[str] = []
    for name in KERNELS:
        launches = capture[name]
        keys = [key for key in launches[sorted(launches, key=_launch_order)[0]]]
        metrics = {key: min(record[key] for record in launches.values()) for key in keys}
        regs = metrics["registers_per_thread"]
        theoretical = metrics["theoretical_occupancy_pct"]
        achieved = metrics["achieved_occupancy_pct"]
        ratio = achieved / theoretical
        if regs > regs_stop:
            verdict = "STOP_REGISTER_HEADROOM"
            why = (
                f"registers/thread {regs:.0f} > {regs_stop:.0f}: the register file "
                f"binds at {metrics['block_limit_registers']:.0f} block(s)/SM; a higher "
                "occupancy target would spill"
            )
        elif ratio >= achieved_ratio:
            verdict = "STOP_NO_GAP"
            why = (
                f"achieved {achieved:.1f}% >= {achieved_ratio:.2f} x theoretical "
                f"{theoretical:.1f}%: at the structural occupancy ceiling"
            )
        else:
            verdict = "OPEN_GAP"
            why = (
                f"achieved {achieved:.1f}% < {achieved_ratio:.2f} x theoretical "
                f"{theoretical:.1f}% and registers/thread {regs:.0f} <= {regs_stop:.0f}"
            )
        report["kernels"][name] = {
            **metrics,
            "launch_count": len(launches),
            "launches": {launch: launches[launch] for launch in sorted(launches, key=_launch_order)},
            "achieved_over_theoretical": round(ratio, 4),
            "verdict": verdict,
            "why": why,
        }
        verdicts.append(verdict)

    if any(v == "OPEN_GAP" for v in verdicts):
        report["decision"] = "OPEN_GAP"
    elif all(v == "STOP_REGISTER_HEADROOM" for v in verdicts):
        report["decision"] = "STOP_REGISTER_HEADROOM"
    else:
        report["decision"] = "STOP_NO_GAP"
    report["reason"] = "; ".join(report["kernels"][name]["why"] for name in KERNELS)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ncu-csv", type=Path, required=True,
                    help="CSV written by ncu --csv (use - for stdin)")
    ap.add_argument("--regs-stop", type=float, default=96.0)
    ap.add_argument("--achieved-ratio", type=float, default=0.8)
    ap.add_argument("--expected-launches", type=int, default=None,
                    help="launch records the capture must hold (ncu --launch-count)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    text = sys.stdin.read() if str(args.ncu_csv) == "-" else args.ncu_csv.read_text()
    try:
        capture = parse_csv(text)
        report = judge(capture, regs_stop=args.regs_stop,
                       achieved_ratio=args.achieved_ratio,
                       expected_launches=args.expected_launches)
    except ValueError as exc:
        report = {"decision": "ABORT", "reason": str(exc)}
    payload = json.dumps(report, indent=2)
    if args.out is not None:
        # The artifact is written even on ABORT: a failed capture is evidence too.
        args.out.write_text(payload + "\n")
    print(payload)
    return 0 if report["decision"] != "ABORT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
