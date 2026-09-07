#!/usr/bin/env python3
"""Classify live spec-decode counters without treating healthy 1.000/7.0 as collapse.

Kit PR #70's gate fails whenever pos0 acceptance ratio is ~1.00. That is the
vllm#53030 LENGTH=1 signature only when later draft positions also collapse to
the same ~1.00 ratio (or never fire) and accepted drafts/step sit near 1. This
cluster's structured bench is 1.000/7.000: every position stays ~1.00 because
the drafter is being accepted, not because graphs pinned a 1-token verify.

A real graph-vs-eager comparison is a separately guarded, restarted/rewarmed
arm. This probe only decides whether that arm is justified.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from typing import Any


SPEC_LINE = re.compile(r"^(vllm:spec_decode_[A-Za-z0-9_]+)(?:\{([^}]*)\})?\s+(\S+)$")
LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')
# Production EXTRA_ARGS for dflash at C4. Token-batch capture covering
# 1..4 seqs × the native eight-row DFlash2 verify, not a 1,2,3,... seq list.
K7_C4_CAPTURE = (1, 2, 4, 8, 16, 24, 32)


def _float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"non-numeric prometheus sample: {value!r}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"invalid prometheus sample: {value!r}")
    return parsed


def parse_metrics(text: str) -> dict[str, Any]:
    drafts = 0.0
    draft_tokens = 0.0
    accepted = 0.0
    pos: dict[int, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = SPEC_LINE.match(line)
        if not match:
            continue
        name, labels, value = match.group(1), match.group(2) or "", match.group(3)
        if name.endswith("_created"):
            continue
        sample = _float(value)
        if name == "vllm:spec_decode_num_drafts_total":
            drafts += sample
        elif name == "vllm:spec_decode_num_draft_tokens_total":
            draft_tokens += sample
        elif name == "vllm:spec_decode_num_accepted_tokens_total":
            accepted += sample
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            fields = dict(LABEL.findall(labels))
            if "position" not in fields:
                raise ValueError(f"per-pos sample missing position: {line}")
            pos[int(fields["position"])] = pos.get(int(fields["position"]), 0.0) + sample
    return {
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted": accepted,
        "pos": pos,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def classify(
    snapshot: dict[str, Any],
    *,
    min_drafts: int = 100,
    k: int = 7,
    capture_sizes: tuple[int, ...] | None = K7_C4_CAPTURE,
    enforce_eager: bool = False,
    argv_known: bool = True,
) -> dict[str, Any]:
    if k < 1:
        raise ValueError("k must be >= 1")
    drafts = float(snapshot.get("drafts") or 0.0)
    draft_tokens = float(snapshot.get("draft_tokens") or 0.0)
    accepted = float(snapshot.get("accepted") or 0.0)
    pos_map = {int(key): float(value) for key, value in (snapshot.get("pos") or {}).items()}
    positions = sorted(pos_map)
    pos_ratios = [
        _ratio(pos_map.get(index, 0.0), drafts) if drafts > 0 else None
        for index in range(k)
    ]
    accepted_fraction = _ratio(accepted, draft_tokens)
    accepted_drafts_per_step = _ratio(accepted, drafts)
    output_tokens_per_step = (
        1.0 + accepted_drafts_per_step if accepted_drafts_per_step is not None else None
    )
    sizes = tuple(int(size) for size in capture_sizes) if capture_sizes is not None else None
    capture_ok = sizes == K7_C4_CAPTURE if sizes is not None else None
    pin_signature = False
    if drafts >= min_drafts and pos_ratios and all(ratio is not None for ratio in pos_ratios):
        later = pos_ratios[1:]
        pin_signature = (
            pos_ratios[0] is not None
            and pos_ratios[0] >= 0.999
            and accepted_drafts_per_step is not None
            and accepted_drafts_per_step <= 1.05
            and (
                not later
                or all(ratio is not None and abs(ratio - pos_ratios[0]) <= 0.01 for ratio in later)
                or all(ratio is not None and ratio <= 0.02 for ratio in later)
            )
        )
    ceiling = (
        drafts >= min_drafts
        and accepted_drafts_per_step is not None
        and accepted_drafts_per_step >= k * 0.98
        and pos_ratios
        and all(ratio is not None and ratio >= 0.98 for ratio in pos_ratios)
    )
    decay = (
        drafts >= min_drafts
        and pos_ratios
        and pos_ratios[0] is not None
        and pos_ratios[-1] is not None
        and pos_ratios[0] < 0.999
        and pos_ratios[-1] + 0.02 < pos_ratios[0]
        and not pin_signature
    )

    if drafts < min_drafts:
        decision = "skip"
        reason = (
            f"only {int(drafts)} drafts since boot (<{min_drafts}); "
            "send a structured or mixed bench first"
        )
        graph_vs_eager = False
    elif pin_signature:
        decision = "collapse"
        reason = (
            "vllm#53030 LENGTH=1 signature: pos0 ~1.00 with later positions "
            "collapsed and accepted drafts/step near 1. A read-only /metrics "
            "scrape is not a graph-vs-eager comparison."
        )
        graph_vs_eager = True
    elif ceiling:
        decision = "healthy-ceiling"
        reason = (
            f"structured-path ceiling: accepted drafts/step ~{k}.0 with every "
            "position ~1.00. This is not collapse and not #54374 "
            "acceptance-LENGTH=1."
        )
        graph_vs_eager = False
    elif decay:
        decision = "healthy-decay"
        reason = (
            "mixed-traffic decay: pos0 is below 1.00 and later positions fall. "
            "Do not restart into ENFORCE_EAGER from this shape."
        )
        graph_vs_eager = False
    else:
        decision = "inconclusive"
        reason = (
            "counters are populated but match neither collapse nor a known "
            "healthy shape. Inspect capture sizes and do not treat pos0=1.00 "
            "alone as a graph bug."
        )
        graph_vs_eager = False

    if not argv_known:
        graph_vs_eager = False
        if decision == "collapse":
            reason += (
                " Live argv was not verified; withhold the eager arm until "
                "capture sizes and --enforce-eager are read from the target."
            )
    elif enforce_eager:
        graph_vs_eager = False
        if decision == "collapse":
            reason += " Live argv is already --enforce-eager; no graph arm remains."

    return {
        "decision": decision,
        "graph_vs_eager_arm_justified": graph_vs_eager,
        "argv_known": bool(argv_known),
        "reason": reason,
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted": accepted,
        "k": k,
        "min_drafts": min_drafts,
        "accepted_fraction": accepted_fraction,
        "accepted_drafts_per_step": accepted_drafts_per_step,
        "output_tokens_per_step": output_tokens_per_step,
        "pos_ratios": pos_ratios,
        "positions": positions,
        "pin_signature": pin_signature,
        "capture_sizes": list(sizes) if sizes is not None else None,
        "capture_sizes_match_k7_c4": capture_ok,
        "enforce_eager": bool(enforce_eager) if argv_known else None,
        "metrics_only": True,
        "quantities": {
            "accepted_fraction": "accepted_tokens / draft_tokens",
            "accepted_drafts_per_step": "accepted_tokens / drafts",
            "output_tokens_per_step": "1 + accepted_drafts_per_step",
            "acceptance_length_1": "collapse to one emitted token per verify, not fraction=1.0",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-file",
        help="Prometheus text. Defaults to stdin.",
    )
    parser.add_argument("--min-drafts", type=int, default=100)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument(
        "--capture-sizes",
        default="",
        help="Live --cudagraph-capture-sizes, comma-separated. Empty = unknown.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Live argv already has --enforce-eager.",
    )
    parser.add_argument(
        "--argv-known",
        action="store_true",
        help="Live PID-1 argv was read from the target (graphs vs eager).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = (
        open(args.metrics_file, encoding="utf-8").read()
        if args.metrics_file
        else __import__("sys").stdin.read()
    )
    size_parts = [part for part in args.capture_sizes.split(",") if part.strip()]
    sizes = tuple(int(part) for part in size_parts) if size_parts else None
    report = classify(
        parse_metrics(text),
        min_drafts=args.min_drafts,
        k=args.k,
        capture_sizes=sizes,
        enforce_eager=args.enforce_eager if args.argv_known else False,
        argv_known=args.argv_known,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["decision"] == "skip":
        return 0
    if report["decision"] == "collapse":
        return 1
    if report["decision"] == "inconclusive":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
