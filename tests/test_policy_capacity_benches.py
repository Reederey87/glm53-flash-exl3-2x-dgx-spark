from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    path = ROOT / "tests" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mixed_prefill_metrics_percentiles_and_rates(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    module = load("bench_mixed_prefill")
    assert module.headers("req") == {
        "Content-Type": "application/json",
        "Authorization": "Bearer secret",
        "X-Request-Id": "req",
    }
    parsed = module.parse_metrics(
        "\n".join(
            [
                'vllm:num_requests_running{model="a"} 1',
                'vllm:num_requests_running{model="b"} 2',
                "vllm:num_requests_waiting 3",
                "vllm:num_preemptions_total 4",
            ]
        )
    )
    assert parsed["num_requests_running"] == 3
    assert parsed["num_requests_waiting"] == 3
    assert module.summarize([4.0, 1.0, 3.0, 2.0]) == {
        "n": 4,
        "median": 2.5,
        "p95": 4.0,
        "p99": 4.0,
        "min": 1.0,
        "max": 4.0,
    }
    events = [
        {"time": 1.0, "count": 1},
        {"time": 2.0, "count": 3},
        {"time": 4.0, "count": 1},
    ]
    assert module.token_rate(events, 1.5, 4.5) == 4 / 3
    assert module.token_arrival_gaps(events, 1.5, 4.5) == [
        0.5,
        0.0,
        0.0,
        2.0,
        0.5,
    ]
    assert module.token_arrival_gaps(events, 2.5, 3.5) == [1.0]


def test_mixed_prefill_context_summary() -> None:
    module = load("bench_mixed_prefill")
    summary = module.summarize_context(
        [
            {
                "newcomer": {"ttft_s": 10.0},
                "incumbent_rate_ratio": 0.95,
                "incumbent_token_arrival_gap_during": {"p99": 0.5},
                "aggregate_tps": 20.0,
                "preemptions": 0,
                "queue_peak_waiting": 1,
            },
            {
                "newcomer": {"ttft_s": 20.0},
                "incumbent_rate_ratio": 0.90,
                "incumbent_token_arrival_gap_during": {"p99": 0.7},
                "aggregate_tps": 18.0,
                "preemptions": 1,
                "queue_peak_waiting": 2,
            },
        ]
    )
    assert summary["newcomer_ttft_s"]["median"] == 15.0
    assert summary["preemptions"] == 1
    assert summary["queue_peak_waiting"] == 2


def test_mixed_prefill_request_validation() -> None:
    module = load("bench_mixed_prefill")
    success = {
        "http": 200,
        "token_events": [{"time": 1.0, "count": 2}],
        "done": True,
        "finish_reason": "length",
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    assert module.request_failure(success) is None
    assert module.request_failure({**success, "done": False}) == "missing SSE [DONE]"
    assert module.request_failure({**success, "error": "boom"}) == "boom"
    cancelled = {
        **success,
        "done": False,
        "finish_reason": None,
        "usage": {},
        "intentional_stop": True,
    }
    assert module.request_failure(
        cancelled, allow_intentional_stop=True
    ) is None


def test_mixed_prefill_cli_fails_closed(
    monkeypatch, tmp_path
) -> None:
    module = load("bench_mixed_prefill")
    monkeypatch.setattr(module, "metrics", lambda: {})
    monkeypatch.setattr(
        module,
        "run_sample",
        lambda *_args, **_kwargs: {
            "context": 9500,
            "sample": 0,
            "error": "newcomer failed",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_mixed_prefill.py",
            "--contexts",
            "9500",
            "--samples",
            "1",
            "--out",
            str(tmp_path / "mixed.json"),
        ],
    )
    assert module.main() == 1


def test_mixed_prefill_orchestration_propagates_stream_error(
    monkeypatch,
) -> None:
    module = load("bench_mixed_prefill")
    monkeypatch.setattr(module, "wait_idle", lambda _timeout: True)
    monkeypatch.setattr(
        module,
        "metrics",
        lambda: {
            "num_requests_running": 0,
            "num_requests_waiting": 0,
            "num_preemptions_total": 0,
            "generation_tokens_total": 0,
        },
    )

    def fake_stream_chat(**kwargs):
        out = kwargs["out"]
        if kwargs.get("first_token_event") is not None:
            now = module.time.perf_counter()
            out.update(
                {
                    "http": 200,
                    "started": now - 1,
                    "ended": now,
                    "token_events": [{"time": now, "count": 2}],
                    "ttft_s": 1.0,
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                    },
                    "finish_reason": "length",
                    "done": True,
                    "intentional_stop": False,
                }
            )
            kwargs["first_token_event"].set()
        else:
            now = module.time.perf_counter()
            out.update(
                {
                    "error": "connection failed",
                    "http": None,
                    "started": now,
                    "ended": now,
                    "token_events": [],
                    "usage": {},
                    "done": False,
                }
            )

    monkeypatch.setattr(module, "stream_chat", fake_stream_chat)
    args = SimpleNamespace(
        idle_timeout=1,
        incumbent_tokens=8,
        timeout=1,
        incumbent_start_timeout=1,
        arrival_delay=0,
        newcomer_tokens=1,
        poll_interval=0.001,
        before_window=1,
    )
    row = module.run_sample(9500, 0, args, "test")
    assert "newcomer: connection failed" in row["error"]
    assert module.summarize_context([row])["valid"] == 0


def test_kv_metrics_and_linear_fit(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    module = load("bench_kv_request_cost")
    parsed = module.parse_metrics(
        "\n".join(
            [
                'vllm:kv_cache_usage_perc{model="a"} 0.1',
                'vllm:kv_cache_usage_perc{model="b"} 0.2',
                "vllm:num_preemptions_total 2",
            ]
        )
    )
    assert parsed["kv_cache_usage_perc"] == 0.30000000000000004
    assert parsed["num_preemptions_total"] == 2
    assert module.fit_line([(0.0, 90_000.0), (10_000.0, 100_000.0), (20_000.0, 110_000.0)]) == {
        "n": 3,
        "intercept": 90000.0,
        "slope": 1.0,
        "r2": 1.0,
    }
    assert module.fit_line([(1.0, 2.0)]) == {
        "n": 1,
        "intercept": None,
        "slope": None,
        "r2": None,
    }


def test_kv_request_validation() -> None:
    module = load("bench_kv_request_cost")
    success = {
        "http": 200,
        "done": True,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    assert module.request_failure(success) is None
    assert module.request_failure({**success, "usage": {}}) == "missing final usage"
    assert module.request_failure({**success, "http": 500}) == "HTTP 500"


def test_kv_cli_fails_closed(monkeypatch, tmp_path) -> None:
    module = load("bench_kv_request_cost")
    monkeypatch.setattr(module, "metrics", lambda: {})
    monkeypatch.setattr(
        module,
        "run_probe",
        lambda *_args, **_kwargs: {
            "context": 2000,
            "rep": 0,
            "error": "request failed",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_kv_request_cost.py",
            "--contexts",
            "2000",
            "--reps",
            "1",
            "--out",
            str(tmp_path / "kv.json"),
        ],
    )
    assert module.main() == 1


def test_kv_orchestration_propagates_stream_error(monkeypatch) -> None:
    module = load("bench_kv_request_cost")
    monkeypatch.setattr(module, "wait_idle", lambda _timeout: True)
    monkeypatch.setattr(
        module,
        "metrics",
        lambda: {
            "kv_cache_usage_perc": 0,
            "num_preemptions_total": 0,
            "num_requests_running": 0,
            "num_requests_waiting": 0,
        },
    )

    def fake_stream_request(*args):
        args[-1].update(
            {
                "error": "connection failed",
                "http": None,
                "usage": {},
                "done": False,
            }
        )

    monkeypatch.setattr(module, "stream_request", fake_stream_request)
    args = SimpleNamespace(
        idle_timeout=1,
        max_tokens=1,
        timeout=1,
        poll_interval=0.001,
        pool_tokens=1_396_551,
    )
    row = module.run_probe(2000, 0, args, "test")
    assert row["error"] == "connection failed"
