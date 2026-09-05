from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "tests" / "bench_decode.py"


def _module(monkeypatch, api_key: str = ""):
    monkeypatch.setenv("VLLM_API_KEY", api_key)
    spec = importlib.util.spec_from_file_location("bench_decode_under_test", BENCH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bearer_header_is_used_for_json_and_metrics(monkeypatch) -> None:
    module = _module(monkeypatch, "secret-for-test")
    assert module._headers(json_content=True) == {
        "Content-Type": "application/json",
        "Authorization": "Bearer secret-for-test",
    }
    assert module._headers() == {"Authorization": "Bearer secret-for-test"}


def test_nan_detection_is_token_aware(monkeypatch) -> None:
    module = _module(monkeypatch)
    for text in ("nan", "value: NaN", "locklock", "the result is (NAN)."):
        assert module.contains_nan(text), text
    for text in ("banana", "finance", "Nanjing", "nanosecond", "canonical"):
        assert not module.contains_nan(text), text


def test_aggregate_any_nan_semantics(monkeypatch) -> None:
    module = _module(monkeypatch)
    clean_runs = [{"nan": module.contains_nan("banana")}, {"nan": False}]
    bad_runs = clean_runs + [{"nan": module.contains_nan("value NaN")}]
    assert not any(run["nan"] for run in clean_runs)
    assert any(run["nan"] for run in bad_runs)
