#!/usr/bin/env python3
"""Compare complete preregistered MTP matrices; never make an adoption decision.

Checks the full 96-cell, three-repetition grid and matching workload controls.
Reports the existing >2% regression-in-two-reps rule, plus baseline spread.
Functional, image, checkpoint, KV and runtime-treatment evidence is separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

LENGTHS = [8192, 16384, 32768, 65536, 131072]
CONCURRENCIES = [1, 2, 4, 8]
WORKLOADS = ["code", "prose", "structured"]
CONTROLS = {
    "model": "nvidia/GLM-5.3-Flash-FP4-Local",
    "prefill_lengths": LENGTHS, "concurrencies": CONCURRENCIES,
    "repetitions": 3, "decode_tokens": 200, "max_seqs": 4,
    "decode_workloads": WORKLOADS, "decode_warmups": 1,
    "decode_chat_template_kwargs": {"reasoning_effort": "max"},
    "decode_token_scope": "generated_tokens_including_reasoning",
    "timing_source": "continuous_completion_usage",
    "prompt_seed_scheme": "sha256-json-tuple-v1", "expect_speculation": True,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value, label: str, *, positive=False) -> float:
    require(type(value) in (float, int) and math.isfinite(value), f"invalid {label}")
    require(value > 0 if positive else value >= 0, f"out-of-range {label}")
    return float(value)


def same_number(actual, expected: float, label: str) -> None:
    require(math.isclose(number(actual, label), expected, rel_tol=1e-9, abs_tol=1e-9),
            f"{label} disagrees with raw evidence")


def key(cell: dict) -> tuple:
    return (cell["kind"], cell["workload"], cell["concurrency"],
            cell["length"] if cell["kind"] == "prefill" else 0, cell["repetition"])


def validate_cell(cell: dict) -> None:
    count = cell["concurrency"]
    require(cell["ok"] == cell["requests"] == count and cell["errors"] == 0
            and not cell["validation_errors"], f"failed cell {key(cell)}")
    results = cell["request_results"]
    hashes = cell["prompt_sha256"]
    require(len(results) == len(hashes) == count, "missing request evidence")
    require(all(isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h)
                for h in hashes), "invalid prompt hash")
    prefill = cell["kind"] == "prefill"
    rates = []
    for result in results:
        require(result["ok"] is True and not result["error"], "failed request")
        budget = 1 if prefill else 200
        require(type(result["completion_tokens"]) is int and result["completion_tokens"] == budget,
                "wrong completion budget")
        prompt = number(result["prompt_tokens"], "prompt tokens", positive=True)
        if prefill:
            require(prompt == cell["length"], "wrong prefill length")
        ttft = number(result["ttft_s"], "TTFT", positive=True)
        arrivals = result["token_arrivals"]
        require(bool(arrivals) and arrivals[-1][0] == budget, "missing token progress")
        previous_count, previous_time = 0, 0.0
        for current_count, elapsed in arrivals:
            require(type(current_count) is int and previous_count < current_count <= budget,
                    "invalid cumulative token count")
            number(elapsed, "arrival time", positive=True)
            require(elapsed >= previous_time, "token arrivals out of order")
            previous_count, previous_time = current_count, elapsed
        same_number(ttft, arrivals[0][1], "TTFT")
        same_number(result["decode_s"], arrivals[-1][1] - arrivals[0][1], "decode interval")
        rates.append(prompt / ttft if prefill else 199 / number(result["decode_s"], "decode interval", positive=True))
    metric = "prefill_tokens_per_s_median" if prefill else "decode_tokens_per_s_median"
    same_number(cell[metric], statistics.median(rates), metric)
    wall = number(cell["wall_s"], "cell duration", positive=True)
    aggregate = sum(r["prompt_tokens"] for r in results) / wall if prefill else count * 199 / wall
    same_number(cell[metric.replace("median", "aggregate")], aggregate, "aggregate throughput")
    spec = cell["acceptance"]
    require(not spec["error"], "invalid speculation counter window")
    accepted, drafted, drafts = (number(spec[name], name) for name in ("accepted", "drafted", "drafts"))
    require(accepted <= drafted and drafts <= drafted, "inconsistent speculation counters")
    if prefill:
        require(cell["prefix_cache_hits"] == 0, "cached prefill")
    else:
        require(drafts > 0 and drafted > 0, "no draft evidence")
        same_number(spec["mean_acceptance_length"], 1 + accepted / drafts, "acceptance length")
        same_number(spec["draft_acceptance_rate"], accepted / drafted, "draft acceptance rate")


def validate(receipt: dict) -> dict:
    require(receipt["complete"] is True and receipt["ok"] is True, "incomplete or failed matrix")
    for name, expected in CONTROLS.items():
        require(receipt.get(name) == expected, f"unregistered control: {name}")
    expected = {("prefill", "random", c, length, rep)
                for c in CONCURRENCIES for length in LENGTHS for rep in (1, 2, 3)}
    expected |= {("decode", workload, c, 0, rep)
                 for c in CONCURRENCIES for workload in WORKLOADS for rep in (1, 2, 3)}
    cells = receipt["cells"]
    require(len(cells) == len(expected) and {key(c) for c in cells} == expected,
            "incomplete or duplicate measured grid")
    warm_expected = Counter({("prefill", "random", c, LENGTHS[0], 0): 1 for c in CONCURRENCIES})
    warm_expected.update({k: 1 for k in expected if k[0] == "decode"})
    require(Counter(key(c) for c in receipt["warmup_cells"]) == warm_expected, "missing or duplicate warmups")
    for cell in cells + receipt["warmup_cells"]:
        validate_cell(cell)
    measured = {key(c): c for c in cells}
    for warm in receipt["warmup_cells"]:
        if warm["kind"] == "decode":
            require(warm["prompt_sha256"] == measured[key(warm)]["prompt_sha256"],
                    "decode warmup does not match measured prompts")
    return measured


def compare(control: dict, candidate: dict) -> dict:
    left, right = validate(control), validate(candidate)
    for name in ("seed", "decode_tasks"):
        require(name in control and control[name] == candidate.get(name), f"mismatched {name}")
    for identity, cell in left.items():
        require(cell["prompt_sha256"] == right[identity]["prompt_sha256"], f"mismatched prompts: {identity}")
    require({key(c): c["prompt_sha256"] for c in control["warmup_cells"]}
            == {key(c): c["prompt_sha256"] for c in candidate["warmup_cells"]},
            "mismatched warmup prompts")
    rows = []
    for group in sorted({identity[:-1] for identity in left}):
        prefix = "prefill" if group[0] == "prefill" else "decode"
        metrics = [f"{prefix}_tokens_per_s_median", f"{prefix}_tokens_per_s_aggregate"]
        if prefix == "decode":
            metrics += ["draft_acceptance_rate", "mean_acceptance_length"]
        for metric in metrics:
            def value(c):
                return c["acceptance"][metric] if metric in ("draft_acceptance_rate", "mean_acceptance_length") else c[metric]
            a = [value(left[(*group, rep)]) for rep in (1, 2, 3)]
            b = [value(right[(*group, rep)]) for rep in (1, 2, 3)]
            require(all(x > 0 for x in a), f"zero control metric: {group} {metric}")
            deltas = [100 * (y / x - 1) for x, y in zip(a, b)]
            median_delta = 100 * (statistics.median(b) / statistics.median(a) - 1)
            spread = 100 * (max(a) - min(a)) / statistics.median(a)
            rows.append({"kind": group[0], "workload": group[1], "concurrency": group[2],
                         "prompt_length": group[3] or None, "metric": metric,
                         "control_reps": a, "candidate_reps": b, "paired_change_pct": deltas,
                         "median_change_pct": median_delta, "control_range_pct": spread,
                         "effect_within_control_range": abs(median_delta) <= spread,
                         "regression_in_two_reps": metric != "mean_acceptance_length" and sum(y < x * 0.98 for x, y in zip(a, b)) >= 2,
                         "gain_at_least_two_pct_in_two_reps": sum(y >= x * 1.02 for x, y in zip(a, b)) >= 2})
    return {"matched_complete_matrices": True, "measured_cells_per_arm": len(left),
            "robust_speed_or_acceptance_regression": any(r["regression_in_two_reps"] for r in rows),
            "adoption_decision": "not made: review control spread, functional checks, frozen runtime and KV evidence",
            "acceptance_length_note": "diagnostic only; changing MTP depth changes its maximum",
            "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.control.read_text()), json.loads(args.candidate.read_text()))
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
