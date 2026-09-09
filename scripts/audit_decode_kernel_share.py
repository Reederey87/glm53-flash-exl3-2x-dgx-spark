#!/usr/bin/env python3
"""Decode-step kernel-share oracle for tasks 29 and 31.

Input is the per-rank chrome trace written by vLLM's in-process torch profiler
(``--profiler-config.profiler=torch``). Kineto reports kernels that run *inside*
a captured CUDA graph as normal ``cat="kernel"`` events, with ``graph id`` /
``graph node id`` and per-launch launch-geometry plus an estimated achieved
occupancy. That is the counter path available on this cluster: external nsys
attach to the live CUDA-graph server is unsafe, ncu replay fail-closes on the
TP2 graph stack (PR #40), and both ncu and nsys GPU-metrics counters are
privilege-denied here (ERR_NVGPUCTRPERM).

This script reports facts and applies pre-registered thresholds. It does not
claim a measured hardware counter and it never treats Kineto's *estimated*
occupancy as a measurement. Kineto reports ``est. achieved occupancy %: 0`` for
every kernel whose ``occupancy.blockLimitSharedMem`` is 0, which is a
calculation artifact (the register/SMEM footprint is reported correctly); the
audit therefore derives occupancy from the launch geometry it can trust
(``warps per SM`` against the block-limit-derived SM warp capacity).

Task 29 gate (fused ``exl3_moe`` decode specialization):
  share of decode-step kernel time < ``--moe-share-floor``
      -> STOP_SHARE_BELOW_FLOOR  (no kernel change can pay end-to-end)
  share >= floor and the weight-streaming model already demands at least
  ``--roofline-tolerance`` of ``--roofline-gbs`` at every target T
      -> STOP_ROOFLINE_BOUND  (routed-expert weight streaming, not occupancy)
  share >= floor and derived (or, failing that, Kineto) occupancy < floor
      -> GAP_CANDIDATE_UNMEASURED  (needs a measured counter; parked here)
  share >= floor and occupancy >= floor
      -> STOP_NO_OCCUPANCY_GAP

Task 31 gate (sparse-MLA captured tactics): the trace cannot carry the runtime
tile config, so the verdict is advisory and keyed on kernel-name diversity and
the sparse-MLA share of decode-step kernel time.

Per-generation attribution: the profiler emits one ``gpu_user_annotation`` range
per captured graph execution, named ``execute_context_<pid>(<rank>)_generation_<seqs>(<tokens>)``.
Joining each kernel into its innermost enclosing range recovers the realized
adaptive-k token count (T) per step, which is what the task 29 gate asks for.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

# Kernel-name classifiers. Order matters: first match wins.
FAMILIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fused_moe", re.compile(r"exl3_moe", re.I)),
    ("grouped_fat_moe", re.compile(r"(fm_gateup|fm_down|fm_gather)", re.I)),
    ("sparse_mla", re.compile(r"(sparse.*mla|mla.*sparse|flashinfer.*mla)", re.I)),
    ("kda", re.compile(r"(fused_recurrent_kda|chunk_kda|causal_conv1d|delta_rule)", re.I)),
    ("attention", re.compile(r"(flash_attn|fmha|attention|paged_attention|flash_fwd)", re.I)),
    ("gemm", re.compile(r"(cutlass|gemm|sgemm|matmul|nvjet|cublas)", re.I)),
    ("collective", re.compile(r"nccl", re.I)),
)

TRACE_SUFFIXES = (".pt.trace.json", ".pt.trace.json.gz")
GENERATION = re.compile(r"_generation_(\d+)\((\d+)\)")
READ_CHUNK = 1 << 20
COMPACT_AT = 1 << 20


def classify(name: str) -> str:
    for family, pattern in FAMILIES:
        if pattern.search(name):
            return family
    return "other"


@dataclass(frozen=True)
class Roofline:
    """Weight-streaming model for the routed-expert path.

    ``expert_bytes_per_rank`` is the per-expert-per-layer payload on one TP rank
    (safetensors headers, ``scripts/audit_live_weight_bytes.py``); ``gbs`` is the
    node's achievable unified-memory bandwidth. Both are model inputs, not
    hardware counters.
    """

    gbs: float
    expert_bytes_per_rank: int
    moe_layers: int
    n_experts: int = 288
    topk: int = 8
    target_t: tuple[int, ...] = (12, 20, 32)
    tolerance: float = 0.85


def iter_trace_events(path: Path) -> Iterator[dict[str, Any]]:
    """Stream ``traceEvents`` without materialising the whole JSON document.

    Handles both minified and pretty-printed traces (the 93 MiB gzip traces
    decompress to ~2.2 GB and ~38 M lines; ``json.load`` on them is unusable).
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    decoder = json.JSONDecoder()
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        buf = ""
        while '"traceEvents"' not in buf:
            chunk = fh.read(READ_CHUNK)
            if not chunk:
                return
            buf += chunk
        buf = buf[buf.index('"traceEvents"'):]
        while True:
            bracket = buf.find("[")
            if bracket != -1:
                buf = buf[bracket + 1:]
                break
            chunk = fh.read(READ_CHUNK)
            if not chunk:
                return
            buf += chunk
        pos = 0
        while True:
            size = len(buf)
            while pos < size and buf[pos] in " \t\r\n,":
                pos += 1
            if pos >= size:
                chunk = fh.read(READ_CHUNK)
                if not chunk:
                    return
                buf = buf[pos:] + chunk
                pos = 0
                continue
            if buf[pos] == "]":
                return
            try:
                event, end = decoder.raw_decode(buf, pos)
            except ValueError:
                chunk = fh.read(READ_CHUNK)
                if not chunk:
                    return
                buf = buf[pos:] + chunk
                pos = 0
                continue
            yield event
            pos = end
            if pos > COMPACT_AT:
                buf = buf[pos:]
                pos = 0


def _grid_volume(args: dict[str, Any]) -> int | None:
    grid = args.get("grid")
    if not isinstance(grid, list) or not grid:
        return None
    total = 1
    for value in grid:
        if not isinstance(value, int) or value <= 0:
            return None
        total *= value
    return total


def _derived_occupancy(args: dict[str, Any]) -> float | None:
    """Occupancy from launch geometry: warps resident / SM warp capacity.

    SM warp capacity comes from the kernel's own ``blockLimitWarps`` (how many
    blocks the warp limit allows) times the warps per block.
    """
    warps_per_sm = args.get("warps per SM")
    block = args.get("block")
    occ = args.get("occupancy")
    if not isinstance(warps_per_sm, (int, float)) or warps_per_sm <= 0:
        return None
    if not isinstance(block, list) or not block or not isinstance(block[0], int):
        return None
    block_limit = occ.get("blockLimitWarps") if isinstance(occ, dict) else None
    if not isinstance(block_limit, int) or block_limit <= 0:
        return None
    warps_per_block = max(1, block[0] // 32)
    capacity = block_limit * warps_per_block
    if capacity <= 0:
        return None
    return round(100.0 * float(warps_per_sm) / capacity, 2)


def find_traces(trace_dir: Path) -> list[Path]:
    found = [
        p
        for p in sorted(trace_dir.rglob("*"))
        if p.is_file() and any(str(p).endswith(sfx) for sfx in TRACE_SUFFIXES)
    ]
    if not found:
        raise FileNotFoundError(f"no *{TRACE_SUFFIXES[0]} traces under {trace_dir}")
    return found


def rank_of(path: Path) -> str:
    match = re.search(r"(?:^|[_.])(rank\d+|worker\d+)", path.name)
    return match.group(1) if match else path.name


def _empty_kernel(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "family": classify(name),
        "count": 0,
        "total_us": 0.0,
        "regs": set(),
        "smem": set(),
        "grid": set(),
        "block": set(),
        "occ_pct": set(),
        "derived_occ_pct": set(),
        "occ_limiter": set(),
        "graphs": set(),
    }


def _attribute_generations(
    annotations: list[tuple[float, float, str]],
    kernels: list[tuple[float, float, str]],
) -> dict[str, dict[str, Any]]:
    """Join each kernel into its innermost enclosing graph-execution range."""
    import heapq

    annotations.sort()
    kernels.sort()
    heap: list[tuple[float, float, str]] = []
    index = 0
    out: dict[str, dict[str, Any]] = {}
    for ts, dur, name in kernels:
        while index < len(annotations) and annotations[index][0] <= ts:
            start, length, label = annotations[index]
            heapq.heappush(heap, (start + length, start, label))
            index += 1
        while heap and heap[0][0] < ts:
            heapq.heappop(heap)
        if not heap:
            continue
        # innermost containing range = latest start among the still-active ones
        _, _, label = max(heap, key=lambda item: item[1])
        bucket = out.setdefault(label, {"kernel_us": 0.0, "kernel_count": 0, "kernels": {}})
        bucket["kernel_us"] += dur
        bucket["kernel_count"] += 1
        row = bucket["kernels"].setdefault(name, {"count": 0, "total_us": 0.0})
        row["count"] += 1
        row["total_us"] += dur
    return out


def _generation_summary(
    attributed: dict[str, dict[str, Any]],
    annotations: list[tuple[float, float, str]],
    roofline: Roofline | None,
) -> dict[str, Any]:
    pieces: dict[str, int] = {}
    for _start, _dur, label in annotations:
        pieces[label] = pieces.get(label, 0) + 1
    out: dict[str, Any] = {}
    for label, bucket in sorted(attributed.items()):
        match = GENERATION.search(label)
        tokens = int(match.group(2)) if match else None
        seqs = int(match.group(1)) if match else None
        families: dict[str, float] = {}
        fused_calls = 0
        fused_us = 0.0
        for name, row in bucket["kernels"].items():
            fam = classify(name)
            families[fam] = round(families.get(fam, 0.0) + row["total_us"], 3)
            if fam == "fused_moe":
                fused_calls += row["count"]
                fused_us += row["total_us"]
        entry: dict[str, Any] = {
            "pieces": pieces.get(label, 0),
            "kernel_count": bucket["kernel_count"],
            "kernel_us": round(bucket["kernel_us"], 3),
            "tokens": tokens,
            "seqs": seqs,
            "families_us": families,
            "fused_moe_us": round(fused_us, 3),
            "fused_moe_calls": fused_calls,
            "fused_moe_share": round(fused_us / bucket["kernel_us"], 6) if bucket["kernel_us"] else 0.0,
        }
        if roofline is not None and fused_calls:
            steps = fused_calls / roofline.moe_layers
            entry["decode_steps"] = round(steps, 3)
            entry["fused_moe_us_per_step"] = round(fused_us / steps, 3)
            if tokens:
                unique = roofline.n_experts * (
                    1.0 - (1.0 - roofline.topk / roofline.n_experts) ** tokens
                )
                entry["expert_slots"] = tokens * roofline.topk
                entry["experts_uniform_estimate"] = round(unique, 2)
                streamed_bytes = fused_calls * unique * roofline.expert_bytes_per_rank
                entry["implied_gbs_uniform"] = round(
                    streamed_bytes / (fused_us / 1e6) / 1e9, 2
                )
                entry["experts_at_roofline"] = round(
                    roofline.gbs * 1e9 * (fused_us / 1e6)
                    / (fused_calls * roofline.expert_bytes_per_rank),
                    2,
                )
        out[label] = entry
    return out


def audit(
    traces: list[Path],
    moe_share_floor: float,
    occ_floor: float,
    roofline: Roofline | None = None,
) -> dict[str, Any]:
    per_kernel: dict[str, dict[str, Any]] = {}
    per_graph: dict[str, dict[str, Any]] = {}
    annotations: list[tuple[float, float, str]] = []
    kernels_index: list[tuple[float, float, str]] = []
    total_kernel_us = 0.0
    total_kernel_count = 0
    total_memcpy_us = 0.0
    per_trace: list[dict[str, Any]] = []
    graphs: set[str] = set()

    for path in traces:
        trace_kernels = 0
        trace_us = 0.0
        for event in iter_trace_events(path):
            cat = event.get("cat")
            if cat == "gpu_user_annotation":
                name = event.get("name")
                if isinstance(name, str) and isinstance(event.get("dur"), (int, float)):
                    annotations.append((float(event["ts"]), float(event["dur"]), name))
                continue
            if cat == "gpu_memcpy" and isinstance(event.get("dur"), (int, float)):
                total_memcpy_us += float(event["dur"])
                continue
            if cat != "kernel" or not isinstance(event.get("dur"), (int, float)):
                continue
            name = str(event.get("name", "<unnamed>"))
            dur = float(event["dur"])
            args = event.get("args") or {}
            trace_kernels += 1
            trace_us += dur
            total_kernel_us += dur
            total_kernel_count += 1
            kernels_index.append((float(event.get("ts", 0.0)), dur, name))
            row = per_kernel.get(name)
            if row is None:
                row = per_kernel[name] = _empty_kernel(name)
            row["count"] += 1
            row["total_us"] += dur
            if isinstance(args.get("registers per thread"), int):
                row["regs"].add(args["registers per thread"])
            if isinstance(args.get("shared memory"), int):
                row["smem"].add(args["shared memory"])
            volume = _grid_volume(args)
            if volume is not None:
                row["grid"].add(volume)
            block = args.get("block")
            if isinstance(block, list) and block:
                row["block"].add(tuple(int(v) for v in block))
            if isinstance(args.get("est. achieved occupancy %"), int):
                row["occ_pct"].add(args["est. achieved occupancy %"])
            derived = _derived_occupancy(args)
            if derived is not None:
                row["derived_occ_pct"].add(derived)
            occ = args.get("occupancy")
            if isinstance(occ, dict) and occ.get("limitingFactors"):
                row["occ_limiter"].add(str(occ["limitingFactors"]))
            gid = args.get("graph id")
            if gid is not None:
                key = str(gid)
                row["graphs"].add(key)
                graphs.add(key)
                graph = per_graph.setdefault(
                    key, {"graph_id": key, "kernel_count": 0, "total_us": 0.0, "kernels": {}}
                )
                graph["kernel_count"] += 1
                graph["total_us"] += dur
                graph["kernels"][name] = graph["kernels"].get(name, 0.0) + dur
        per_trace.append(
            {
                "file": str(path),
                "rank": rank_of(path),
                "kernel_events": trace_kernels,
                "kernel_us": round(trace_us, 3),
            }
        )

    if not total_kernel_count:
        raise ValueError("no kernel events in the supplied traces")

    kernels_out = []
    for row in per_kernel.values():
        share = row["total_us"] / total_kernel_us if total_kernel_us else 0.0
        kernels_out.append(
            {
                "name": row["name"],
                "family": row["family"],
                "count": row["count"],
                "total_us": round(row["total_us"], 3),
                "mean_us": round(row["total_us"] / row["count"], 4),
                "share": round(share, 6),
                "regs": sorted(row["regs"]),
                "smem": sorted(row["smem"]),
                "grid_volume": sorted(row["grid"]),
                "block": [list(b) for b in sorted(row["block"])],
                "est_occupancy_pct": sorted(row["occ_pct"]),
                "derived_occupancy_pct": sorted(row["derived_occ_pct"]),
                "occupancy_limiter": sorted(row["occ_limiter"]),
                "graph_ids": sorted(row["graphs"]),
            }
        )
    kernels_out.sort(key=lambda r: r["total_us"], reverse=True)

    families: dict[str, dict[str, Any]] = {}
    for row in kernels_out:
        fam = families.setdefault(row["family"], {"total_us": 0.0, "count": 0, "kernels": []})
        fam["total_us"] = round(fam["total_us"] + row["total_us"], 3)
        fam["count"] += row["count"]
        fam["kernels"].append(row["name"])
    for fam in families.values():
        fam["share"] = round(fam["total_us"] / total_kernel_us, 6) if total_kernel_us else 0.0

    graphs_out = []
    for graph in per_graph.values():
        top = sorted(graph["kernels"].items(), key=lambda kv: kv[1], reverse=True)[:8]
        graphs_out.append(
            {
                "graph_id": graph["graph_id"],
                "kernel_count": graph["kernel_count"],
                "total_us": round(graph["total_us"], 3),
                "top": [{"name": n, "total_us": round(v, 3)} for n, v in top],
            }
        )
    graphs_out.sort(key=lambda g: g["total_us"], reverse=True)

    attributed = _attribute_generations(annotations, kernels_index)
    generations = _generation_summary(attributed, annotations, roofline)

    fused = families.get("fused_moe", {"total_us": 0.0, "share": 0.0, "kernels": []})
    fused_kernels = [r for r in kernels_out if r["family"] == "fused_moe"]
    occ_values = [v for r in fused_kernels for v in r["est_occupancy_pct"]]
    min_occ = min(occ_values) if occ_values else None
    derived_values = [v for r in fused_kernels for v in r["derived_occupancy_pct"]]
    min_derived_occ = min(derived_values) if derived_values else None

    roofline_out: dict[str, Any] | None = None
    roofline_bound = False
    if roofline is not None:
        by_t: dict[str, Any] = {}
        implied: list[float] = []
        for tokens in roofline.target_t:
            calls = 0
            us = 0.0
            for entry in generations.values():
                if entry.get("tokens") == tokens:
                    calls += entry["fused_moe_calls"]
                    us += entry["fused_moe_us"]
            if not calls or us <= 0:
                by_t[str(tokens)] = {"measured": False}
                continue
            steps = calls / roofline.moe_layers
            unique = roofline.n_experts * (
                1.0 - (1.0 - roofline.topk / roofline.n_experts) ** tokens
            )
            streamed = calls * unique * roofline.expert_bytes_per_rank
            implied_gbs = streamed / (us / 1e6) / 1e9
            at_roofline = (
                roofline.gbs * 1e9 * (us / 1e6)
                / (calls * roofline.expert_bytes_per_rank)
            )
            by_t[str(tokens)] = {
                "measured": True,
                "fused_moe_calls": calls,
                "decode_steps": round(steps, 3),
                "fused_moe_us_per_step": round(us / steps, 3),
                "expert_slots": tokens * roofline.topk,
                "experts_uniform_estimate": round(unique, 2),
                "experts_at_roofline": round(at_roofline, 2),
                "implied_gbs_uniform": round(implied_gbs, 2),
            }
            implied.append(implied_gbs)
        measured = [v for v in by_t.values() if v.get("measured")]
        roofline_bound = bool(measured) and len(measured) == len(roofline.target_t) and min(implied) >= (
            roofline.gbs * roofline.tolerance
        )
        roofline_out = {
            "gbs": roofline.gbs,
            "tolerance": roofline.tolerance,
            "expert_bytes_per_rank": roofline.expert_bytes_per_rank,
            "moe_layers": roofline.moe_layers,
            "n_experts": roofline.n_experts,
            "topk": roofline.topk,
            "target_t": list(roofline.target_t),
            "by_t": by_t,
            "all_target_t_measured": len(measured) == len(roofline.target_t),
            "roofline_bound": roofline_bound,
            "note": (
                "weight-streaming model (safetensors headers + achievable bandwidth), "
                "not a hardware counter; experts_uniform_estimate is an upper bound "
                "on the unique routed experts per layer"
            ),
        }

    if fused["share"] < moe_share_floor:
        task29 = {
            "verdict": "STOP_SHARE_BELOW_FLOOR",
            "reason": (
                f"fused exl3_moe share {fused['share']:.4f} < floor {moe_share_floor:.4f}; "
                "no decode-kernel specialization can pay end-to-end"
            ),
        }
    elif roofline_bound:
        task29 = {
            "verdict": "STOP_ROOFLINE_BOUND",
            "reason": (
                f"fused exl3_moe share {fused['share']:.4f} >= floor {moe_share_floor:.4f}, "
                "but the routed-expert weight-streaming model already demands "
                f">= {roofline.tolerance:.0%} of {roofline.gbs:.0f} GB/s at every target T; "
                "the kernel is memory-bound, not occupancy/GEMV-bound"
            ),
        }
    elif min_derived_occ is not None and min_derived_occ < occ_floor:
        task29 = {
            "verdict": "GAP_CANDIDATE_UNMEASURED",
            "reason": (
                f"fused share {fused['share']:.4f} >= floor {moe_share_floor:.4f} and "
                f"derived occupancy {min_derived_occ}% < {occ_floor}%; a measured "
                "counter is still required before any kernel change"
            ),
        }
    elif min_occ is None:
        task29 = {
            "verdict": "GAP_CANDIDATE_UNMEASURED",
            "reason": "fused share above floor but no occupancy data in the trace",
        }
    elif min_occ < occ_floor:
        task29 = {
            "verdict": "GAP_CANDIDATE_UNMEASURED",
            "reason": (
                f"fused share {fused['share']:.4f} >= floor {moe_share_floor:.4f} and "
                f"estimated occupancy {min_occ}% < {occ_floor}%; a measured counter is "
                "still required before any kernel change"
            ),
        }
    else:
        task29 = {
            "verdict": "STOP_NO_OCCUPANCY_GAP",
            "reason": (
                f"fused share {fused['share']:.4f} but occupancy {min_occ}% >= {occ_floor}%"
            ),
        }
    task29.update(
        {
            "fused_moe_us": fused["total_us"],
            "fused_moe_share": fused["share"],
            "fused_kernels": fused["kernels"],
            "est_occupancy_pct_min": min_occ,
            "derived_occupancy_pct_min": min_derived_occ,
            "moe_share_floor": moe_share_floor,
            "occ_floor": occ_floor,
            "measured_counter_available": False,
        }
    )

    sparse = [r for r in kernels_out if r["family"] == "sparse_mla"]
    names = sorted({r["name"] for r in sparse})
    decode_names = sorted(
        {
            r["name"]
            for r in sparse
            if "decode" in r["name"].lower() and "merge" not in r["name"].lower()
        }
    )
    sparse_us = sum(r["total_us"] for r in sparse)
    if len(decode_names) <= 1:
        verdict31 = "STOP_NO_TACTIC_DIVERSITY"
    else:
        verdict31 = "AUDIT_TACTIC_DIVERSITY"
    task31 = {
        "sparse_mla_kernels": [
            {
                "name": r["name"],
                "count": r["count"],
                "total_us": r["total_us"],
                "mean_us": r["mean_us"],
                "share": r["share"],
                "block": r["block"],
                "grid_volume": r["grid_volume"],
                "graph_ids": r["graph_ids"],
            }
            for r in sparse
        ],
        "sparse_mla_share": round(sparse_us / total_kernel_us, 6) if total_kernel_us else 0.0,
        "distinct_names": len(names),
        "distinct_decode_names": len(decode_names),
        "decode_kernel_names": decode_names,
        "verdict": verdict31,
        "advisory": True,
        "note": (
            "kernel name does not encode the runtime tile config; a single decode "
            "kernel name across every captured q means the heuristic picked one "
            "tactic, not that the tactic is optimal. sparse_mla_share bounds the "
            "end-to-end payoff of any tactic change."
        ),
    }

    return {
        "schema": 2,
        "source": "vllm torch profiler (in-process, CUDA-graph aware)",
        "traces": per_trace,
        "totals": {
            "kernel_us": round(total_kernel_us, 3),
            "kernel_count": total_kernel_count,
            "memcpy_us": round(total_memcpy_us, 3),
            "distinct_graphs": len(graphs),
            "distinct_kernels": len(kernels_out),
            "graph_execution_ranges": len(annotations),
        },
        "families": families,
        "kernels": kernels_out,
        "graphs": graphs_out,
        "generations": generations,
        "roofline": roofline_out,
        "decision": {"task29": task29, "task31": task31},
    }


def render(report: dict[str, Any], top: int) -> str:
    lines: list[str] = []
    totals = report["totals"]
    lines.append(
        f"kernel_us={totals['kernel_us']:.0f} kernels={totals['kernel_count']} "
        f"graphs={totals['distinct_graphs']} distinct={totals['distinct_kernels']} "
        f"ranges={totals['graph_execution_ranges']}"
    )
    lines.append("families:")
    for fam, row in sorted(report["families"].items(), key=lambda kv: kv[1]["total_us"], reverse=True):
        lines.append(f"  {fam:16s} {row['share']*100:6.2f}%  {row['total_us']:12.0f} us  n={row['count']}")
    lines.append(f"top {top} kernels:")
    for row in report["kernels"][:top]:
        occ = ",".join(str(v) for v in row["est_occupancy_pct"]) or "-"
        der = ",".join(str(v) for v in row["derived_occupancy_pct"]) or "-"
        lines.append(
            f"  {row['share']*100:6.2f}%  {row['total_us']:11.0f} us  n={row['count']:<6d} "
            f"occ%={occ:<6s} derived%={der:<6s} regs={row['regs']} smem={row['smem']}  "
            f"{row['name'][:88]}"
        )
    if report["generations"]:
        lines.append("graph executions by realized T:")
        for label, row in sorted(
            report["generations"].items(), key=lambda kv: kv[1]["kernel_us"], reverse=True
        ):
            extra = ""
            if "fused_moe_us_per_step" in row:
                extra = (
                    f" steps={row['decode_steps']:.0f} moe_ms/step="
                    f"{row['fused_moe_us_per_step']/1000:.2f} implied_GB/s={row['implied_gbs_uniform']:.0f}"
                )
            lines.append(
                f"  {label:36s} {row['kernel_us']/1e6:7.2f}s  fused={row['fused_moe_share']*100:5.2f}%"
                f"  T={row['tokens']}{extra}"
            )
    if report.get("roofline"):
        roof = report["roofline"]
        lines.append(
            f"roofline: {roof['gbs']:.0f} GB/s, {roof['expert_bytes_per_rank']} B/expert/layer/rank, "
            f"{roof['moe_layers']} MoE layers -> bound={roof['roofline_bound']}"
        )
        for tokens, row in sorted(roof["by_t"].items(), key=lambda kv: int(kv[0])):
            if not row.get("measured"):
                lines.append(f"  T={tokens}: not measured")
                continue
            lines.append(
                f"  T={tokens}: slots={row['expert_slots']} uniform_E={row['experts_uniform_estimate']} "
                f"E_at_roofline={row['experts_at_roofline']} implied={row['implied_gbs_uniform']:.0f} GB/s"
            )
    lines.append("graphs (by time):")
    for graph in report["graphs"][:10]:
        lines.append(
            f"  graph {graph['graph_id']:>5s}  {graph['total_us']:11.0f} us  n={graph['kernel_count']}"
        )
    d29 = report["decision"]["task29"]
    lines.append(
        f"task29: {d29['verdict']}  fused_share={d29['fused_moe_share']*100:.2f}% "
        f"est_occ_min={d29['est_occupancy_pct_min']} derived_occ_min={d29['derived_occupancy_pct_min']}"
    )
    lines.append(f"  reason: {d29['reason']}")
    d31 = report["decision"]["task31"]
    lines.append(
        f"task31: {d31['verdict']}  distinct_sparse_mla={d31['distinct_names']} "
        f"decode_names={d31['distinct_decode_names']} share={d31['sparse_mla_share']*100:.2f}%"
    )
    for row in d31["sparse_mla_kernels"][:10]:
        lines.append(f"  {row['share']*100:6.2f}%  n={row['count']:<6d} {row['name'][:88]}")
    return "\n".join(lines)


def parse_roofline(args: argparse.Namespace) -> Roofline | None:
    if not args.roofline_gbs:
        return None
    if not args.expert_bytes_per_rank or not args.moe_layers:
        raise ValueError(
            "--roofline-gbs needs --expert-bytes-per-rank and --moe-layers"
        )
    return Roofline(
        gbs=args.roofline_gbs,
        expert_bytes_per_rank=args.expert_bytes_per_rank,
        moe_layers=args.moe_layers,
        n_experts=args.n_experts,
        topk=args.topk,
        target_t=tuple(int(t) for t in args.target_t.split(",")),
        tolerance=args.roofline_tolerance,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace-dir", type=Path)
    src.add_argument("--trace", type=Path, action="append", default=[])
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--moe-share-floor", type=float, default=0.05)
    ap.add_argument("--occ-floor", type=float, default=50.0)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--roofline-gbs", type=float, default=0.0,
                    help="node achievable unified-memory bandwidth (0 disables the model)")
    ap.add_argument("--expert-bytes-per-rank", type=int, default=0,
                    help="per-expert per-layer payload bytes on one TP rank")
    ap.add_argument("--moe-layers", type=int, default=0)
    ap.add_argument("--n-experts", type=int, default=288)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--target-t", default="12,20,32")
    ap.add_argument("--roofline-tolerance", type=float, default=0.85)
    args = ap.parse_args(argv)

    try:
        traces = find_traces(args.trace_dir) if args.trace_dir else list(args.trace)
        for path in traces:
            if not path.is_file():
                raise FileNotFoundError(path)
        report = audit(traces, args.moe_share_floor, args.occ_floor, parse_roofline(args))
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"decode-share audit FAILED: {exc}", file=sys.stderr)
        return 2

    print(render(report, args.top))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=1) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
