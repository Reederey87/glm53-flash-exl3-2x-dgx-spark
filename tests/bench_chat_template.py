#!/usr/bin/env python3
"""Compare two GLM chat templates on ordinary and fallback render paths."""

import argparse
import json
import statistics
import time
from pathlib import Path

from jinja2 import Environment


ROOT = Path(__file__).parents[1]


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name


def compile_template(path: Path):
    return Environment(extensions=["jinja2.ext.loopcontrols"]).from_string(
        path.read_text()
    )


def tool_call(index: int) -> dict[str, object]:
    call_id = f"call-{index}"
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "lookup", "arguments": {"query": call_id}},
    }


def valid_messages(count: int) -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call(index) for index in range(count)],
        },
        *[
            {
                "role": "tool",
                "tool_call_id": f"call-{index}",
                "content": f"result-{index}",
            }
            for index in reversed(range(count))
        ],
    ]


def fallback_messages(count: int) -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call(index) for index in range(count)],
        },
        {"role": "tool", "content": "missing-id"},
        *[
            {
                "role": "tool",
                "tool_call_id": f"call-{index}",
                "content": f"result-{index}",
            }
            for index in range(1, count)
        ],
    ]


def fallback_list_messages(count: int) -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call(index) for index in range(count)],
        },
        {
            "role": "tool",
            "content": [
                {"output": "missing-id"},
                *[
                    {
                        "tool_call_id": f"call-{index}",
                        "output": f"result-{index}",
                    }
                    for index in range(1, count)
                ],
            ],
        },
    ]


def render(template, messages: list[dict[str, object]]) -> str:
    return template.render(
        messages=messages,
        tools=None,
        add_generation_prompt=False,
    )


def sample(template, messages: list[dict[str, object]], iterations: int) -> float:
    started = time.perf_counter()
    for _ in range(iterations):
        render(template, messages)
    return time.perf_counter() - started


def benchmark(
    control,
    candidate,
    messages: list[dict[str, object]],
    *,
    rounds: int,
    iterations: int,
) -> dict[str, float]:
    control_output = render(control, messages)
    candidate_output = render(candidate, messages)
    if control_output != candidate_output:
        raise RuntimeError("control and candidate outputs differ in benchmark case")

    render(control, messages)
    render(candidate, messages)
    control_samples: list[float] = []
    candidate_samples: list[float] = []
    for round_index in range(rounds):
        if round_index % 2:
            candidate_samples.append(sample(candidate, messages, iterations))
            control_samples.append(sample(control, messages, iterations))
        else:
            control_samples.append(sample(control, messages, iterations))
            candidate_samples.append(sample(candidate, messages, iterations))

    control_median = statistics.median(control_samples)
    candidate_median = statistics.median(candidate_samples)
    ratio = candidate_median / control_median
    return {
        "control_seconds": control_median,
        "candidate_seconds": candidate_median,
        "candidate_over_control": ratio,
        "improvement": 1.0 - ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "files" / "chat_template.jinja",
    )
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    control = compile_template(args.control)
    candidate = compile_template(args.candidate)
    cases = {
        "ordinary_valid_10": valid_messages(10),
        "fallback_missing_id_500": fallback_messages(500),
        "fallback_list_missing_id_500": fallback_list_messages(500),
    }
    results = {
        name: benchmark(
            control,
            candidate,
            messages,
            rounds=args.rounds,
            iterations=args.iterations,
        )
        for name, messages in cases.items()
    }
    ordinary_ok = results["ordinary_valid_10"]["candidate_over_control"] <= 1.05
    fallback_ok = all(
        result["improvement"] >= 0.15
        for name, result in results.items()
        if name.startswith("fallback_")
    )
    report = {
        "control": display_path(args.control),
        "candidate": display_path(args.candidate),
        "rounds": args.rounds,
        "iterations": args.iterations,
        "results": results,
        "gates": {
            "ordinary_regression_at_most_5_percent": ordinary_ok,
            "fallback_improvement_at_least_15_percent": fallback_ok,
        },
        "passed": ordinary_ok and fallback_ok,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
