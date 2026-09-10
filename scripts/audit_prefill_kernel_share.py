#!/usr/bin/env python3
"""Prefill kernel-share oracle for the E3 W4-successor gate (task 24).

Why this exists
---------------
``docs/11`` §8 ranks the W4 successor (a persistent SUH-scaled A-cache consumed
read-only by every gate/up K-slab) TEST-NEXT behind an ncu measurement of the
gather-vs-gateup share of E3 layer time. ncu is not usable on this cluster: it
is not installed and the driver enforces ``RmProfilingAdminOnly: 1``, so both
ncu and ``nsys --gpu-metrics`` fail closed (``ERR_NVGPUCTRPERM``). The counter
path that *is* safe and available here is the per-rank chrome trace written by
vLLM's in-process torch profiler. Kineto records every grouped ``fm_*`` kernel
with its duration and launch geometry, including launches that happen inside a
captured CUDA graph, so the *share* question can be answered without hardware
counters.

What it reports
---------------
Per rank (two ranks have independent GPU clock domains, so they are summarised
separately and never summed):

* total kernel time and family shares;
* the three grouped fat-MoE kernels -- ``fm_gather_kernel`` (gather + input
  Hadamard into ``h13``), ``fm_gateup_kernel``, ``fm_down_kernel`` -- as shares
  of total kernel time and of E3 layer time (their sum);
* the launch geometry the trace carries (registers, SMEM, block, grid volume)
  and the derived occupancy, which is context for W5, not a measurement;
* the decode share, so a capture that is not prefill-dominated fails closed.

Pre-registered gate (2026-09-09, before the cluster window)
-----------------------------------------------------------
Let ``g`` = ``fm_gather_kernel`` time / (gather + gateup + down) time and
``t`` = ``fm_gather_kernel`` time / total kernel time, on a rank. Take the
minimum of each across ranks (conservative). All three grouped kernels must be
present with positive time on every audited rank; otherwise the capture is
``INCONCLUSIVE_NO_GROUPED_CAPTURE`` (a lone gather kernel would otherwise read
as a 100% share on no evidence).

* ``g >= --gather-share-floor`` (default 0.10) **and**
  ``t >= --gather-total-floor`` (default 0.03)
      -> ``PROCEED_W4_SUCCESSOR``: removing the gather can plausibly clear the
      standing >=3% end-to-end bar, so the kernel work is worth opening.
* otherwise
      -> ``STOP_GATHER_SHARE_BELOW_FLOOR``: the ceiling is too small. W4
      already measured a -10.7% regression at 60k for the fused-gather rewrite,
      so a low gather share does not justify another attempt.

Rationale for the floors: the successor can at best remove the gather kernel;
a 10% share of E3 layer time means even a partial recovery can move
end-to-end, and the 3% total-time floor matches the W5 stop bar.

Prefill-vs-decode guard (calibrated 2026-09-09 on this deployment)
-----------------------------------------------------------------
The fused ``exl3_moe`` kernel is *not* decode-only here: with
``EXL3_FAT_GROUPED=1``/``EXL3_TEMP_ROWS_FUSED=32`` a prefill chunk runs one
fused launch per layer beside the grouped triple. Measured on the first 60k
capture, on both ranks: 1470 fused launches against 1428 grouped launches per
rank (ratio 1.03) and a fused share of 12.7% of kernel time, while the profiler's
own step table attributes 96.8% of CUDA time to 1792-token context steps and
1.7% to a final 896-token context step -- i.e. essentially the whole window is
prefill. A decode-dominated capture instead measures 50.7% fused (PR #64). The
guard therefore uses both a time ceiling (``--decode-share-max``, default 0.30:
2.3x above the measured prefill value and 1.7x below the measured decode value)
and a structural launch ratio (``--fused-call-ratio-max``, default 2.0: 1.9x
above the measured prefill ratio, and unreachable for a capture that runs the
fused path every step).

Both guards are calibrated heuristics on this pinned window (one cold prefill,
``max_tokens=1``), not general phase identification: the launch ratio
approximates ``1 + fused-only steps / grouped steps`` rather than decode *time*,
and a mixed workload with enough concurrent decode can land between the two
measured regimes. They stop a window whose capture is not the prefill it claims
to be; they do not certify a share under arbitrary concurrent traffic.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_decode_kernel_share as base  # noqa: E402

# First match wins. Restricted to names the base auditor already classifies as
# grouped_fat_moe, so a generic "down"/"gather" token cannot match.
ROLES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("gather", re.compile(r"fm_gather", re.I)),
    ("gateup", re.compile(r"fm_gateup", re.I)),
    ("down", re.compile(r"fm_down", re.I)),
)

VERDICT_PROCEED = "PROCEED_W4_SUCCESSOR"
VERDICT_STOP = "STOP_GATHER_SHARE_BELOW_FLOOR"
VERDICT_NO_GROUPED = "INCONCLUSIVE_NO_GROUPED_CAPTURE"
VERDICT_DECODE = "INCONCLUSIVE_DECODE_DOMINATED"


def role_of(name: str) -> str | None:
    if base.classify(name) != "grouped_fat_moe":
        return None
    for role, pattern in ROLES:
        if pattern.search(name):
            return role
    return None


def summarize_rank(report: dict[str, Any], top: int) -> dict[str, Any]:
    """Reduce one rank's ``audit_decode_kernel_share.audit`` report to shares."""
    totals = report["totals"]
    total_us = float(totals["kernel_us"])
    grouped: dict[str, dict[str, Any]] = {}
    e3_us = 0.0
    for row in report["kernels"]:
        role = role_of(str(row["name"]))
        if role is None:
            continue
        e3_us += float(row["total_us"])
        grouped[role] = {
            "name": row["name"],
            "count": row["count"],
            "total_us": row["total_us"],
            "share_of_total": row["share"],
            "share_of_e3": None,  # filled below once the E3 total is known
            "regs": row["regs"],
            "smem": row["smem"],
            "block": row["block"],
            "grid_volume": row["grid_volume"],
            "derived_occupancy_pct": row["derived_occupancy_pct"],
        }
    for row in grouped.values():
        row["share_of_e3"] = (
            round(float(row["total_us"]) / e3_us, 6) if e3_us else None
        )
    families = {
        fam: {"total_us": body["total_us"], "count": body["count"], "share": body["share"]}
        for fam, body in report["families"].items()
    }
    return {
        "kernel_us": round(total_us, 3),
        "kernel_count": totals["kernel_count"],
        "memcpy_us": totals["memcpy_us"],
        "distinct_kernels": totals["distinct_kernels"],
        "e3_us": round(e3_us, 3),
        "e3_share_of_total": round(e3_us / total_us, 6) if total_us else 0.0,
        "grouped": grouped,
        "families": families,
        "kernels": report["kernels"][:top],
    }


def decide(
    ranks: dict[str, dict[str, Any]],
    gather_share_floor: float,
    gather_total_floor: float,
    decode_share_max: float,
    fused_call_ratio_max: float = 2.0,
) -> dict[str, Any]:
    if not ranks:
        return {"verdict": VERDICT_NO_GROUPED, "reason": "no ranks in the audit"}
    missing = sorted(r for r, body in ranks.items() if not body["grouped"])
    if missing:
        return {
            "verdict": VERDICT_NO_GROUPED,
            "reason": f"rank(s) {missing} captured no grouped fm_* kernels",
            "gather_share_floor": gather_share_floor,
            "gather_total_floor": gather_total_floor,
            "decode_share_max": decode_share_max,
            "fused_call_ratio_max": fused_call_ratio_max,
        }
    # A gather-vs-E3 share is only meaningful when the whole E3 layer ran: a
    # trace with one lone fm_gather_kernel would otherwise read as a 100%
    # gather share and pass the gate on no evidence.
    incomplete: dict[str, list[str]] = {}
    for rank, body in ranks.items():
        absent = [
            role
            for role in ("gather", "gateup", "down")
            if role not in body["grouped"] or float(body["grouped"][role]["total_us"]) <= 0
        ]
        if absent:
            incomplete[rank] = absent
    if incomplete:
        return {
            "verdict": VERDICT_NO_GROUPED,
            "reason": (
                "an E3 layer share needs all three grouped kernels with positive "
                f"time; missing roles by rank: {incomplete}"
            ),
            "gather_share_floor": gather_share_floor,
            "gather_total_floor": gather_total_floor,
            "decode_share_max": decode_share_max,
            "fused_call_ratio_max": fused_call_ratio_max,
        }
    decode_share = {
        r: float(body["families"].get("fused_moe", {}).get("share", 0.0))
        for r, body in ranks.items()
    }
    fused_ratio = {
        r: (
            float(body["families"].get("fused_moe", {}).get("count", 0))
            / max(1.0, float(body["grouped"]["gather"]["count"]))
        )
        for r, body in ranks.items()
    }
    worst_decode = max(decode_share.values())
    worst_ratio = max(fused_ratio.values())
    if worst_decode > decode_share_max or worst_ratio > fused_call_ratio_max:
        return {
            "verdict": VERDICT_DECODE,
            "reason": (
                f"fused exl3_moe share {worst_decode:.4f} vs max {decode_share_max:.4f} "
                f"and fused/gather launch ratio {worst_ratio:.2f} vs max "
                f"{fused_call_ratio_max:.2f}; this capture is not prefill-dominated"
            ),
            "decode_share_by_rank": decode_share,
            "decode_share_max": decode_share_max,
            "fused_call_ratio_by_rank": fused_ratio,
            "fused_call_ratio_max": fused_call_ratio_max,
        }
    share_of_e3 = {
        r: float(body["grouped"]["gather"]["share_of_e3"]) for r, body in ranks.items()
    }
    share_of_total = {
        r: float(body["grouped"]["gather"]["share_of_total"]) for r, body in ranks.items()
    }
    min_e3 = min(share_of_e3.values())
    min_total = min(share_of_total.values())
    proceed = min_e3 >= gather_share_floor and min_total >= gather_total_floor
    reason = (
        f"min gather share of E3 layer time {min_e3:.4f} vs floor "
        f"{gather_share_floor:.4f}; min gather share of total kernel time "
        f"{min_total:.4f} vs floor {gather_total_floor:.4f}"
    )
    if not proceed:
        reason += (
            "; the successor can at best remove the gather, so this ceiling "
            "does not justify opening the kernel window"
        )
    return {
        "verdict": VERDICT_PROCEED if proceed else VERDICT_STOP,
        "reason": reason,
        "gather_share_of_e3_by_rank": share_of_e3,
        "gather_share_of_total_by_rank": share_of_total,
        "min_gather_share_of_e3": round(min_e3, 6),
        "min_gather_share_of_total": round(min_total, 6),
        "gather_share_floor": gather_share_floor,
        "gather_total_floor": gather_total_floor,
        "decode_share_by_rank": decode_share,
        "decode_share_max": decode_share_max,
        "fused_call_ratio_by_rank": fused_ratio,
        "fused_call_ratio_max": fused_call_ratio_max,
        "e2e_ceiling_if_gather_free": round(min_total, 6),
    }


def audit_prefill(
    traces: list[Path],
    gather_share_floor: float = 0.10,
    gather_total_floor: float = 0.03,
    decode_share_max: float = 0.30,
    fused_call_ratio_max: float = 2.0,
    top: int = 15,
) -> dict[str, Any]:
    ranks: dict[str, dict[str, Any]] = {}
    for path in traces:
        report = base.audit([path], moe_share_floor=0.05, occ_floor=50.0)
        ranks[base.rank_of(path)] = summarize_rank(report, top)
    return {
        "schema": 1,
        "source": "vllm torch profiler (in-process, CUDA-graph aware)",
        "gate": {
            "name": "w4_successor_gather_share",
            "preregistered": "2026-09-09, before the cluster window",
            "gather_share_floor": gather_share_floor,
            "gather_total_floor": gather_total_floor,
            "decode_share_max": decode_share_max,
            "fused_call_ratio_max": fused_call_ratio_max,
        },
        "traces": [str(p) for p in traces],
        "ranks": ranks,
        "decision": decide(
            ranks, gather_share_floor, gather_total_floor, decode_share_max, fused_call_ratio_max
        ),
    }


def render(report: dict[str, Any], top: int) -> str:
    lines: list[str] = []
    for rank, body in sorted(report["ranks"].items()):
        lines.append(
            f"[{rank}] kernel_us={body['kernel_us']:.0f} kernels={body['kernel_count']} "
            f"e3_us={body['e3_us']:.0f} e3_share={body['e3_share_of_total']*100:.2f}% "
            f"memcpy_us={body['memcpy_us']:.0f}"
        )
        lines.append("  families:")
        for fam, row in sorted(
            body["families"].items(), key=lambda kv: kv[1]["total_us"], reverse=True
        ):
            lines.append(
                f"    {fam:16s} {row['share']*100:6.2f}%  {row['total_us']:12.0f} us  "
                f"n={row['count']}"
            )
        lines.append("  E3 grouped breakdown:")
        for role in ("gather", "gateup", "down"):
            row = body["grouped"].get(role)
            if row is None:
                lines.append(f"    {role:7s} missing")
                continue
            der = ",".join(str(v) for v in row["derived_occupancy_pct"]) or "-"
            lines.append(
                f"    {role:7s} {row['share_of_total']*100:6.2f}% of total  "
                f"{row['share_of_e3']*100:6.2f}% of E3  {row['total_us']:11.0f} us  "
                f"n={row['count']:<6d} derived%={der:<6s} regs={row['regs']} "
                f"smem={row['smem']} block={row['block']} grid={row['grid_volume']}"
            )
        lines.append(f"  top {top} kernels:")
        for row in body["kernels"][:top]:
            der = ",".join(str(v) for v in row["derived_occupancy_pct"]) or "-"
            lines.append(
                f"    {row['share']*100:6.2f}%  {row['total_us']:11.0f} us  "
                f"n={row['count']:<6d} derived%={der:<6s} {row['name'][:78]}"
            )
    dec = report["decision"]
    lines.append(f"decision: {dec['verdict']}")
    lines.append(f"  {dec['reason']}")
    if "gather_share_of_e3_by_rank" in dec:
        lines.append(
            "  gather share of E3 by rank: "
            + ", ".join(f"{r}={v*100:.2f}%" for r, v in sorted(dec["gather_share_of_e3_by_rank"].items()))
        )
        lines.append(
            "  gather share of total by rank: "
            + ", ".join(f"{r}={v*100:.2f}%" for r, v in sorted(dec["gather_share_of_total_by_rank"].items()))
        )
        lines.append(
            f"  ceiling if the gather became free: "
            f"{dec['e2e_ceiling_if_gather_free']*100:.2f}% of prefill kernel time"
        )
    if "fused_call_ratio_by_rank" in dec:
        lines.append(
            "  fused/gather launch ratio by rank: "
            + ", ".join(
                f"{r}={v:.2f}" for r, v in sorted(dec["fused_call_ratio_by_rank"].items())
            )
            + f" (max {dec['fused_call_ratio_max']:.2f})"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace-dir", type=Path)
    src.add_argument("--trace", type=Path, action="append", default=[])
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--gather-share-floor", type=float, default=0.10)
    ap.add_argument("--gather-total-floor", type=float, default=0.03)
    ap.add_argument("--decode-share-max", type=float, default=0.30,
                    help="fused exl3_moe share of kernel time; measured prefill 0.128, "
                         "measured decode 0.507 (PR #64)")
    ap.add_argument("--fused-call-ratio-max", type=float, default=2.0,
                    help="fused exl3_moe launches / fm_gather launches; measured prefill 1.03")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args(argv)

    try:
        traces = base.find_traces(args.trace_dir) if args.trace_dir else list(args.trace)
        for path in traces:
            if not path.is_file():
                raise FileNotFoundError(path)
        report = audit_prefill(
            traces,
            args.gather_share_floor,
            args.gather_total_floor,
            args.decode_share_max,
            args.fused_call_ratio_max,
            args.top,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"prefill-share audit FAILED: {exc}", file=sys.stderr)
        return 2

    print(render(report, args.top))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=1) + "\n")
        print(f"wrote {args.json_out}")
    verdict = report["decision"]["verdict"]
    return 0 if verdict in (VERDICT_PROCEED, VERDICT_STOP) else 3


if __name__ == "__main__":
    sys.exit(main())
