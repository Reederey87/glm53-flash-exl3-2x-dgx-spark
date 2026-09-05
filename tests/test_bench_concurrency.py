from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "tests" / "bench_concurrency.py"


def _module(monkeypatch, api_key: str = ""):
    monkeypatch.setenv("VLLM_API_KEY", api_key)
    spec = importlib.util.spec_from_file_location("bench_concurrency_under_test", BENCH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_auth_and_request_id_headers(monkeypatch) -> None:
    module = _module(monkeypatch, "test-key")
    assert module._headers(json_content=True, request_id="req-1") == {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-key",
        "X-Request-Id": "req-1",
    }


def test_metrics_sum_labeled_series_and_busy_veto(monkeypatch) -> None:
    module = _module(monkeypatch)
    parsed = module.parse_metrics(
        "\n".join(
            [
                'vllm:num_requests_running{model_name="a"} 1',
                'vllm:num_requests_running{model_name="b"} 2',
                'vllm:num_requests_waiting{model_name="a"} 0',
                'vllm:prefix_cache_hits_total{model_name="a"} 10',
                'vllm:prefix_cache_hits_total{model_name="b"} 5',
                'vllm:prefix_cache_queries_total{model_name="a"} 20',
                "vllm:num_preemptions_total 3",
            ]
        )
    )
    assert parsed["num_requests_running"] == 3
    assert parsed["prefix_cache_hits_total"] == 15
    assert parsed["num_preemptions_total"] == 3
    assert module.server_is_busy(parsed)
    assert not module.server_is_busy(
        {"num_requests_running": 0, "num_requests_waiting": 0}
    )


def test_cached_prefix_warmup_and_measurement_use_different_suffixes(monkeypatch) -> None:
    module = _module(monkeypatch)
    warm = module.build_messages("code", 0, 100, 2, False, warmup=True)
    measured = module.build_messages("code", 0, 100, 2, False, warmup=False)
    assert warm[0] == measured[0]
    assert warm[-1] != measured[-1]
    shared = module.build_messages("chat", 1, 100, 99, True)
    other_shared = module.build_messages("chat", 1, 100, 3, True)
    assert shared[0] == other_shared[0]


def test_percentiles_and_request_ids_are_stable(monkeypatch) -> None:
    module = _module(monkeypatch)
    assert module.percentiles([]) == {"p50": None, "p95": None, "p99": None}
    assert module.percentiles([4.0, 1.0, 3.0, 2.0]) == {
        "p50": 3.0,
        "p95": 4.0,
        "p99": 4.0,
    }
    assert (
        module._request_id("abc", "code", 60000, 4, 2, 3, "measure")
        == "glm53-ladder-abc-code-c60000-n4-r2-l3-measure"
    )
