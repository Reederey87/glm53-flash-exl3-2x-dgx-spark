from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import io

import pytest


@pytest.fixture
def quality():
    path = Path(__file__).resolve().parents[1] / "scripts/quality_gsm8k.py"
    spec = importlib.util.spec_from_file_location("quality_gsm8k", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response(content="#### 42"):
    return {"model": "nvidia/GLM-5.3-Flash-FP4-Local",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": content, "reasoning": "The distractor answer is 999."}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}}


@pytest.mark.parametrize("text,expected", [("work\n#### 42", "42"), ("#### -1,234.50", "-1234.5"),
    ("#### +0.25\n", "0.25"), ("42", None), ("#### 4\nActually 42", None),
    ("#### 1,23", None), ("#### NaN", None), ("#### 42 dollars", None), (None, None)])
def test_final_answer_extraction(quality, text, expected):
    assert quality.answer(text) == expected


@pytest.mark.parametrize("fault", ["truncated", "empty", "reasoning_only", "bad_usage", "bool_usage", "over_budget", "model", "choices", "tool", "index"])
def test_invalid_responses_cannot_pass(quality, fault):
    body = response()
    if fault == "truncated":
        body["choices"][0]["finish_reason"] = "length"
    elif fault == "empty":
        body["choices"][0]["message"]["content"] = ""
    elif fault == "reasoning_only":
        body["choices"][0]["message"].update(content=None, reasoning="#### 42")
    elif fault == "bad_usage":
        body["usage"]["total_tokens"] = 49
    elif fault == "bool_usage":
        body["usage"]["prompt_tokens"] = True
    elif fault == "over_budget":
        body["usage"].update(completion_tokens=8193, total_tokens=8213)
    elif fault == "model":
        body["model"] = "different"
    elif fault == "choices":
        body["choices"] *= 2
    elif fault == "tool":
        body["choices"][0]["message"]["tool_calls"] = [{"id": "unexpected"}]
    else:
        body["choices"][0]["index"] = 1
    result = quality.score_response(body, "42")
    assert not result["correct"] and result["faults"]


def test_incorrect_answer_and_format_are_scored_not_dropped(quality):
    wrong = quality.score_response(response("#### 999"), "42")
    malformed = quality.score_response(response("Answer: 42"), "42")
    assert not wrong["correct"] and not wrong["faults"]
    assert malformed["format_error"] and not malformed["correct"]
    assert quality.score_response(response("#### 42.0"), "42")["correct"]


def test_request_excludes_reference_and_freezes_reasoning(quality):
    row = {"question": "Compute six times seven.", "answer": "SECRET REFERENCE"}
    request = quality.payload(row)
    assert "SECRET" not in json.dumps(request)
    assert request["chat_template_kwargs"] == {"reasoning_effort": "max"}
    assert request["max_tokens"] == 8192 and request["temperature"] == 0


def test_exact_upper_bound_and_no_gain_cancellation(quality):
    assert quality.loss_upper_bound(0, 10) == pytest.approx(1 - 0.05 ** 0.1)
    # For n=2, k=1, P(X<=1)=1-p**2. Inverting at alpha=.05 gives sqrt(.95).
    assert quality.loss_upper_bound(1, 2) == pytest.approx(0.95 ** 0.5)
    assert quality.loss_upper_bound(2, 2) == 1
    good = quality.score_response(response(), "42")
    bad = quality.score_response(response("#### 0"), "42")
    control = [deepcopy(good) for _ in range(quality.COUNT)]
    candidate = deepcopy(control)
    assert quality.compare(control, candidate)["component_pass"]
    for i in range(20):
        candidate[i] = bad
        control[20 + i] = bad
    result = quality.compare(control, candidate)
    assert result["accuracy_delta_percentage_points"] == 0
    assert len(result["lost_case_ids"]) == len(result["gained_case_ids"]) == 20
    assert not result["component_pass"]


def test_low_baseline_or_unfinished_answer_prevents_pass(quality):
    good = quality.score_response(response(), "42")
    bad = quality.score_response(response("#### 0"), "42")
    assert not quality.compare([bad] * quality.COUNT, [bad] * quality.COUNT)["component_pass"]
    rows = [deepcopy(good) for _ in range(quality.COUNT)]
    broken = deepcopy(rows)
    broken[0]["faults"] = ["unfinished"]
    broken[0]["correct"] = False
    assert not quality.compare(rows, broken)["component_pass"]


@pytest.fixture
def arm(quality, tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "COUNT", 2)
    rows = [{"question": "six times seven?", "answer": "#### 42"},
            {"question": "forty plus two?", "answer": "#### 42"}]
    meta = {"protocol": quality.PROTOCOL, "serving_invocation": "a" * 32,
            "runtime_receipt_sha256": "b" * 64,
            "script_sha256": hashlib.sha256(Path(quality.__file__).read_bytes()).hexdigest()}
    records = [{"id": i, "request_sha256": quality.digest(quality.payload(row)),
                "http_status": 200, "error": "", "response": response()} for i, row in enumerate(rows)]
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    (tmp_path / "complete.json").write_text(json.dumps({"complete": True, "count": 2, "serving_invocation": "a" * 32}))
    (tmp_path / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return quality, tmp_path, rows, records, meta


@pytest.mark.parametrize("fault", ["partial", "duplicate", "prompt", "http", "error", "protocol", "runner", "completion", "provenance"])
def test_receipt_validation_rejects_corruption(arm, fault):
    quality, path, rows, records, meta = arm
    assert len(quality.read_arm(path, rows)) == 2
    if fault == "partial":
        records.pop()
    elif fault == "duplicate":
        records[1] = deepcopy(records[0])
    elif fault == "prompt":
        records[0]["request_sha256"] = "c" * 64
    elif fault == "http":
        records[0]["http_status"] = 500
    elif fault == "error":
        records[0]["error"] = "TimeoutError"
    elif fault == "protocol":
        meta = deepcopy(meta)
        meta["protocol"]["temperature"] = 1
    elif fault == "runner":
        meta["script_sha256"] = "d" * 64
    elif fault == "completion":
        (path / "complete.json").write_text('{"complete":false}')
    else:
        meta.pop("runtime_receipt_sha256")
    (path / "metadata.json").write_text(json.dumps(meta))
    (path / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    with pytest.raises(ValueError):
        quality.read_arm(path, rows)


def test_same_serving_arm_cannot_be_its_own_candidate(arm):
    quality, path, rows, _, _ = arm
    with pytest.raises(ValueError, match="distinct"):
        quality.compare_arms(path, path, rows)


def test_guard_rejects_live_speed_matrix_and_changed_server(quality, monkeypatch):
    def command(args, **kwargs):
        return "ActiveState=active\nInvocationID=" + "a" * 32 + "\n" if "show" in args else "glm53-int64-baseline-matrix.service loaded activating start\n"
    monkeypatch.setattr(quality.subprocess, "check_output", command)
    with pytest.raises(RuntimeError, match="speed matrix"):
        quality.serving_guard("a" * 32)
    with pytest.raises(RuntimeError, match="invocation"):
        quality.serving_guard("b" * 32)


def test_dataset_tampering_rejected_before_requests(quality, tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text('{"question":"x","answer":"#### 42"}\n')
    with pytest.raises(ValueError, match="bytes"):
        quality.corpus(path)


@pytest.mark.parametrize("fail_second", [False, True])
def test_runner_preserves_responses_without_retries(quality, tmp_path, monkeypatch, fail_second):
    monkeypatch.setattr(quality, "COUNT", 2)
    monkeypatch.setattr(quality, "serving_guard", lambda _: None)
    requests = []

    def transport(req, **kwargs):
        requests.append(json.loads(req.data))
        if fail_second and len(requests) == 2:
            raise TimeoutError("do not expose arbitrary exception details")
        stream = io.BytesIO(json.dumps(response()).encode())
        stream.status = 200
        return stream

    monkeypatch.setattr(quality.urllib.request, "urlopen", transport)
    proof = tmp_path / "runtime.json"
    proof.write_text('{}')
    out = tmp_path / "arm"
    rows = [{"question": "six times seven?", "answer": "#### 42"}] * 2
    if fail_second:
        with pytest.raises(RuntimeError, match="no retry"):
            quality.run(rows, out, "a" * 32, proof)
        assert not (out / "complete.json").exists()
    else:
        quality.run(rows, out, "a" * 32, proof)
        assert len(quality.read_arm(out, rows)) == 2
    assert len(requests) == 2
    records = [json.loads(line) for line in (out / "responses.jsonl").read_text().splitlines()]
    assert len(records) == 2 and records[0]["response"] == response()
    assert records[1]["error"] == ("TimeoutError" if fail_second else "")
    with pytest.raises(FileExistsError):
        quality.run(rows, out, "a" * 32, proof)
    assert len(requests) == 2


def test_live_guard_prevents_output_and_network(quality, tmp_path, monkeypatch):
    def refuse(_):
        raise RuntimeError("speed matrix active")

    monkeypatch.setattr(quality, "serving_guard", refuse)
    monkeypatch.setattr(quality.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network must not be called"))
    proof = tmp_path / "runtime.json"
    proof.write_text('{}')
    out = tmp_path / "arm"
    with pytest.raises(RuntimeError, match="matrix"):
        quality.run([], out, "a" * 32, proof)
    assert not out.exists()
