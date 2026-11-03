from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def matrix_pair():
    path = Path(__file__).resolve().parents[1] / "scripts/compare_mtp_matrix.py"
    spec = importlib.util.spec_from_file_location("compare_mtp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def cell(kind, workload, concurrency, length, rep):
        prefill = kind == "prefill"
        first = length / 1000 if prefill else 0.1
        interval = 0.0 if prefill else 1.99
        budget = 1 if prefill else 200
        return {"kind": kind, "workload": workload, "concurrency": concurrency,
                "length": length, "repetition": rep, "ok": concurrency,
                "requests": concurrency, "errors": 0, "validation_errors": [],
                "wall_s": first + interval,
                "prompt_sha256": [hashlib.sha256(f"{kind}-{workload}-{concurrency}-{length}-{rep}-{i}".encode()).hexdigest() for i in range(concurrency)],
                "prefix_cache_hits": 0 if prefill else None,
                "request_results": [{"ok": True, "error": "", "prompt_tokens": length,
                    "completion_tokens": budget, "ttft_s": first, "decode_s": interval,
                    "token_arrivals": [[1, first]] if prefill else [[1, first], [200, first + interval]]}
                    for _ in range(concurrency)],
                f"{kind}_tokens_per_s_median": 1000 if prefill else 100,
                f"{kind}_tokens_per_s_aggregate": concurrency * (length if prefill else 199) / (first + interval),
                "acceptance": {"accepted": 250, "drafted": 500, "drafts": 100,
                    "mean_acceptance_length": 3.5, "draft_acceptance_rate": 0.5, "error": ""}}
    receipt = {**deepcopy(module.CONTROLS), "ok": True, "complete": True,
               "seed": 2026110304, "decode_tasks": {w: w for w in module.WORKLOADS},
               "cells": [], "warmup_cells": []}
    for rep in (1, 2, 3):
        for concurrency in module.CONCURRENCIES:
            for length in module.LENGTHS:
                receipt["cells"].append(cell("prefill", "random", concurrency, length, rep))
            if rep == 1:
                receipt["warmup_cells"].append(cell("prefill", "random", concurrency, 8192, 0))
            for workload in module.WORKLOADS:
                measured = cell("decode", workload, concurrency, 128, rep)
                receipt["cells"].append(measured)
                receipt["warmup_cells"].append(deepcopy(measured))
    return module, receipt, deepcopy(receipt)


def test_mtp_comparison_does_not_turn_lower_acceptance_length_into_regression(matrix_pair):
    module, control, candidate = matrix_pair
    for c in candidate["cells"] + candidate["warmup_cells"]:
        if c["kind"] == "decode":
            c["acceptance"].update(accepted=220, drafted=300, mean_acceptance_length=3.2,
                                   draft_acceptance_rate=220 / 300)
    result = module.compare(control, candidate)
    assert result["measured_cells_per_arm"] == 96
    assert not result["robust_speed_or_acceptance_regression"]
    assert result["adoption_decision"].startswith("not made")
    lengths = [r for r in result["rows"] if r["metric"] == "mean_acceptance_length"]
    assert all(r["median_change_pct"] < 0 and not r["regression_in_two_reps"] for r in lengths)


@pytest.mark.parametrize("reps,accepted,rejected", [(1, 240, False), (2, 240, True), (2, 245, False)])
def test_mtp_comparison_uses_repeated_strict_two_percent_threshold(matrix_pair, reps, accepted, rejected):
    module, control, candidate = matrix_pair
    for c in candidate["cells"]:
        if c["kind"] == "decode" and c["workload"] == "code" and c["concurrency"] == 1 and c["repetition"] <= reps:
            c["acceptance"].update(accepted=accepted, mean_acceptance_length=1 + accepted / 100,
                                   draft_acceptance_rate=accepted / 500)
    assert module.compare(control, candidate)["robust_speed_or_acceptance_regression"] == rejected


@pytest.mark.parametrize("fault", ["incomplete", "missing", "duplicate", "warmup", "cached", "prompt", "budget", "counter", "timing", "effort", "rate"])
def test_mtp_comparison_rejects_insufficient_or_mismatched_evidence(matrix_pair, fault):
    module, control, candidate = matrix_pair
    if fault == "incomplete":
        candidate["complete"] = False
    elif fault == "missing":
        candidate["cells"].pop()
    elif fault == "duplicate":
        candidate["cells"][-1] = deepcopy(candidate["cells"][0])
    elif fault == "warmup":
        candidate["warmup_cells"].pop()
    elif fault == "cached":
        candidate["cells"][0]["prefix_cache_hits"] = 256
    elif fault == "prompt":
        candidate["cells"][0]["prompt_sha256"][0] = "f" * 64
    elif fault == "budget":
        candidate["cells"][0]["request_results"][0]["completion_tokens"] = 2
    elif fault == "counter":
        candidate["cells"][5]["acceptance"]["draft_acceptance_rate"] = 0.9
    elif fault == "timing":
        candidate["cells"][0]["request_results"][0]["token_arrivals"][0][1] += 1
    elif fault == "effort":
        candidate["decode_chat_template_kwargs"]["reasoning_effort"] = "low"
    else:
        candidate["cells"][0]["prefill_tokens_per_s_median"] *= 2
    with pytest.raises(ValueError):
        module.compare(control, candidate)
