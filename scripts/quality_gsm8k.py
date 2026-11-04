#!/usr/bin/env python3
"""Frozen completed-answer GSM8K component; not a full model-quality gate.

Run only on spark1 in an isolated, qualified serving window. No automatic
retries, partial-run resumption, dataset download or deployment changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
DATA_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
COUNT = 1319
MODEL = "nvidia/GLM-5.3-Flash-FP4-Local"
PROTOCOL = {
    "version": 1, "dataset_revision": REVISION, "dataset_sha256": DATA_SHA256,
    "count": COUNT, "model": MODEL, "concurrency": 1, "max_tokens": 8192,
    "temperature": 0, "top_p": 1, "seed": 2026110401,
    "chat_template_kwargs": {"reasoning_effort": "max"},
    "instruction": "Solve the problem. End your final answer with a separate line: #### <numeric answer>. Do not include units on that line.",
    "minimum_control_accuracy": 0.95, "maximum_loss_upper_bound": 0.01,
    "loss_bound_alpha": 0.05,
}
NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def answer(text: object) -> str | None:
    """Score only a final, standalone numeric answer, never reasoning text."""
    if not isinstance(text, str) or not text.strip():
        return None
    match = re.fullmatch(r"####\s+(" + NUMBER + r")", text.strip().splitlines()[-1].strip())
    if not match:
        return None
    return str(Decimal(match[1].replace(",", "")).normalize())


def corpus(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("dataset bytes do not match the frozen test split")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != COUNT or any(not isinstance(r.get("question"), str) or not r["question"] or answer(r.get("answer")) is None for r in rows):
        raise ValueError("invalid dataset count/question/answer")
    return rows


def payload(row: dict) -> dict:
    return {
        **{k: PROTOCOL[k] for k in ("model", "max_tokens", "temperature", "top_p", "seed", "chat_template_kwargs")},
        "stream": False,
        "messages": [{"role": "user", "content": row["question"] + "\n\n" + PROTOCOL["instruction"]}],
    }


def score_response(body: object, expected: str) -> dict:
    faults = []
    choice = {}
    usage = {}
    if isinstance(body, dict):
        choices = body.get("choices")
        if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
            choice = choices[0]
        else:
            faults.append("choices")
        usage = body.get("usage")
        if body.get("model") != MODEL:
            faults.append("model")
    else:
        faults.append("response")
    message = choice.get("message")
    if not isinstance(message, dict):
        message = {}
    content = message.get("content")
    if message.get("role") != "assistant" or message.get("tool_calls"):
        faults.append("message")
    if choice.get("index") != 0:
        faults.append("choice_index")
    if choice.get("finish_reason") != "stop":
        faults.append("unfinished")
    if not isinstance(content, str) or not content.strip():
        faults.append("empty_final_answer")
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] <= 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
        faults.append("usage")
    elif usage["completion_tokens"] > PROTOCOL["max_tokens"] or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        faults.append("usage")
    predicted = answer(content)
    return {"correct": not faults and predicted is not None and Decimal(predicted) == Decimal(expected),
            "predicted": predicted, "faults": faults, "finish_reason": choice.get("finish_reason"),
            "format_error": predicted is None}


def loss_upper_bound(losses: int, count: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson binomial upper bound (no gain cancellation)."""
    if not 0 <= losses <= count or count <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid binomial inputs")
    if losses == count:
        return 1.0
    if losses == 0:
        return -math.expm1(math.log(alpha) / count)
    lo, hi = 0.0, 1.0
    for _ in range(64):
        p = (lo + hi) / 2
        terms = [math.lgamma(count + 1) - math.lgamma(k + 1) - math.lgamma(count - k + 1)
                 + k * math.log(p) + (count - k) * math.log1p(-p) for k in range(losses + 1)]
        peak = max(terms)
        cdf = math.exp(peak) * sum(math.exp(t - peak) for t in terms)
        if cdf > alpha:
            lo = p
        else:
            hi = p
    return hi


def read_arm(path: Path, rows: list[dict]) -> list[dict]:
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("protocol") != PROTOCOL or meta.get("script_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("protocol or scorer/runner bytes differ")
    if not re.fullmatch(r"[0-9a-f]{32}", meta.get("serving_invocation", "")) or not re.fullmatch(r"[0-9a-f]{64}", meta.get("runtime_receipt_sha256", "")):
        raise ValueError("missing runtime provenance")
    final = json.loads((path / "complete.json").read_text())
    if final != {"complete": True, "count": COUNT, "serving_invocation": meta.get("serving_invocation")}:
        raise ValueError("missing or invalid completion evidence")
    records = [json.loads(line) for line in (path / "responses.jsonl").read_text().splitlines()]
    if len(records) != len(rows):
        raise ValueError("partial or extra responses")
    scores = []
    for i, (record, row) in enumerate(zip(records, rows)):
        if record.get("id") != i or record.get("request_sha256") != digest(payload(row)):
            raise ValueError("reordered/duplicate/mismatched request")
        if record.get("http_status") != 200 or record.get("error"):
            raise ValueError("transport failure")
        scores.append(score_response(record.get("response"), answer(row["answer"])))
    return scores


def compare(control: list[dict], candidate: list[dict]) -> dict:
    if len(control) != COUNT or len(candidate) != COUNT:
        raise ValueError("full test split required")
    a = sum(bool(r["correct"]) for r in control)
    b = sum(bool(r["correct"]) for r in candidate)
    losses = [i for i, (x, y) in enumerate(zip(control, candidate)) if x["correct"] and not y["correct"]]
    gains = [i for i, (x, y) in enumerate(zip(control, candidate)) if not x["correct"] and y["correct"]]
    faults = {arm: {str(i): r["faults"] for i, r in enumerate(scores) if r["faults"]}
              for arm, scores in (("control", control), ("candidate", candidate))}
    upper = loss_upper_bound(len(losses), COUNT, PROTOCOL["loss_bound_alpha"])
    passed = not any(faults.values()) and a / COUNT >= PROTOCOL["minimum_control_accuracy"] and upper <= PROTOCOL["maximum_loss_upper_bound"]
    return {"scope": "GSM8K completed-answer component only; no general quality or adoption claim",
            "count": COUNT, "control_correct": a, "candidate_correct": b,
            "control_accuracy": a / COUNT, "candidate_accuracy": b / COUNT,
            "accuracy_delta_percentage_points": 100 * (b - a) / COUNT,
            "lost_case_ids": losses, "gained_case_ids": gains,
            "loss_probability_upper_bound": upper, "faults": faults,
            "format_errors": {arm: sum(r["format_error"] for r in scores) for arm, scores in (("control", control), ("candidate", candidate))},
            "component_pass": passed}


def serving_guard(invocation: str) -> None:
    state = subprocess.check_output(["systemctl", "--user", "show", "vllm-glm53fp4.service", "-p", "ActiveState", "-p", "InvocationID"], text=True)
    values = dict(line.split("=", 1) for line in state.splitlines() if "=" in line)
    if values.get("ActiveState") != "active" or values.get("InvocationID") != invocation:
        raise RuntimeError("serving invocation changed or is not active")
    units = subprocess.check_output(["systemctl", "--user", "list-units", "--type=service", "--state=active,activating", "--no-legend", "--plain"], text=True)
    if any("matrix" in line.lower() and "glm53" in line.lower() for line in units.splitlines()):
        raise RuntimeError("speed matrix unit is still active; quality requests refused")


def compare_arms(control: Path, candidate: Path, rows: list[dict]) -> dict:
    left = json.loads((control / "metadata.json").read_text())
    right = json.loads((candidate / "metadata.json").read_text())
    if control.resolve() == candidate.resolve() or left.get("serving_invocation") == right.get("serving_invocation"):
        raise ValueError("two distinct qualified serving invocations required")
    return compare(read_arm(control, rows), read_arm(candidate, rows))


def run(rows: list[dict], out: Path, invocation: str, runtime_receipt: Path) -> None:
    # The runtime receipt must be separately qualified; its hash alone is not
    # treatment proof. Keep raw responses so scoring can be independently audited.
    if not re.fullmatch(r"[0-9a-f]{32}", invocation):
        raise ValueError("expected systemd invocation ID")
    runtime_hash = hashlib.sha256(runtime_receipt.read_bytes()).hexdigest()
    serving_guard(invocation)
    out.mkdir(parents=True, exist_ok=False)
    metadata = {"protocol": PROTOCOL, "serving_invocation": invocation,
                "runtime_receipt_sha256": runtime_hash,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "started_at": datetime.now(timezone.utc).isoformat()}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (out / "responses.jsonl").open("x") as output:
        for i, row in enumerate(rows):
            serving_guard(invocation)
            request = payload(row)
            record = {"id": i, "request_sha256": digest(request), "http_status": 0, "error": "", "response": None}
            started = time.monotonic()
            try:
                req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                    data=json.dumps(request).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=1800) as response:
                    record["http_status"] = response.status
                    record["response"] = json.load(response)
            except Exception as exc:
                record["error"] = type(exc).__name__
            record["wall_s"] = time.monotonic() - started
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            if record["error"] or record["http_status"] != 200:
                raise RuntimeError(f"request {i} failed; receipt retained, no retry")
            serving_guard(invocation)
    (out / "complete.json").write_text(json.dumps({"complete": True, "count": COUNT, "serving_invocation": invocation}) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    runner = commands.add_parser("run")
    runner.add_argument("--out", type=Path, required=True)
    runner.add_argument("--serving-invocation", required=True)
    runner.add_argument("--runtime-receipt", type=Path, required=True)
    comparator = commands.add_parser("compare")
    comparator.add_argument("--control", type=Path, required=True)
    comparator.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    rows = corpus(args.dataset)
    if args.command == "prepare":
        print(json.dumps({"protocol": PROTOCOL, "protocol_sha256": digest(PROTOCOL),
                          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "requests_sha256": digest([digest(payload(row)) for row in rows]),
                          "all_reference_answers_parse": True}, indent=2))
    elif args.command == "run":
        run(rows, args.out, args.serving_invocation, args.runtime_receipt)
    else:
        result = compare_arms(args.control, args.candidate, rows)
        print(json.dumps(result, indent=2))
        return 0 if result["component_pass"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
