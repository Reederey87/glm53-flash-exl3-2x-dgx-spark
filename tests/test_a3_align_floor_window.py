from __future__ import annotations

import importlib.util
import json
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_a3_align_floor_window.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_a3_align_floor_window", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def valid_env() -> dict[str, str]:
    return {
        "LONG_PREFILL_TOKEN_THRESHOLD": "3584",
        "MAX_NUM_BATCHED_TOKENS": "3584",
        "MAX_NUM_SEQS": "4",
        "GLM53_ALIGN_FLOOR": "1",
        "GLM53_MIXED_PREFILL_CHUNK": "skip",
        "GLM53_MIXED_PREFILL_MAX_WAIT_MS": "1500",
        "GLM53_MIXED_PREFILL_WARM_TOKENS": "3584",
        "GLM53_MIXED_PREFILL_LATE_CAP": "512",
        "GLM53_MIXED_PREFILL_ESCALATE_MS": "10000",
        "GLM53_MIXED_PREFILL_LATE_CAP_MAX": "1792",
    }


def test_parse_and_validate_effective_arm() -> None:
    module = load_module()
    text = "\n".join(f"{key}={value}" for key, value in valid_env().items())
    parsed = module.parse_env(text)
    assert parsed == valid_env()
    assert module.validate_arm(parsed, "1") == []


def test_validation_rejects_wrong_or_unsafe_arm() -> None:
    module = load_module()
    env = valid_env()
    env["LONG_PREFILL_TOKEN_THRESHOLD"] = "1792"
    env["GLM53_ALIGN_FLOOR"] = "0"
    env["GLM53_MIXED_PREFILL_LATE_CAP"] = "3584"
    env["GLM53_MIXED_PREFILL_CHUNK"] = "512"
    errors = module.validate_arm(env, "1")
    assert any("LPTT" in error for error in errors)
    assert any("late cap" in error for error in errors)
    assert any("arm mismatch" in error for error in errors)
    assert any("must remain skip" in error for error in errors)


def test_extracts_only_sub_page_decode_floor_caps() -> None:
    module = load_module()
    logs = """
[glm53-decode-floor-v3.1] late-admit req=chatcmpl-a waited_ms=1501 remaining=60000 cap=512
[glm53-decode-floor-v3.1] late-escalate req=chatcmpl-a-1a2b3c4d cap=512->1024
[glm53-decode-floor-v3] late-admit req=chatcmpl-unrelated-a-1a2b3c4d waited_ms=1501 remaining=60000 cap=1792
[glm53-decode-floor-v3] late-escalate req=chatcmpl-a-1234567g cap=1024->1792
[glm53-decode-floor-v3] late-escalate req=chatcmpl-a-lookalike cap=1024->1792
[glm53-decode-floor-v3] late-escalate req=chatcmpl-a-abcdef12-extra cap=1024->1792
[glm53-decode-floor-v3] late-escalate req=chatcmpl-a-abcdef12 cap=1792->3584
"""
    assert module.extract_subblock_caps(logs, {"a"}) == [512, 1024]
    assert module.extract_subblock_caps(logs, {"missing"}) == []


def test_scheduler_request_id_matching_is_anchored() -> None:
    module = load_module()
    request_ids = {"glm53-c1-run-c60000-r0-newcomer"}
    raw = "glm53-c1-run-c60000-r0-newcomer"
    assert module.matches_scheduler_request_id(f"chatcmpl-{raw}", request_ids)
    assert module.matches_scheduler_request_id(
        f"chatcmpl-{raw}-1a2b3c4d", request_ids
    )
    for lookalike in (
        raw,
        f"xchatcmpl-{raw}",
        f"chatcmpl-{raw}x",
        f"chatcmpl-{raw}-1a2b3c4",
        f"chatcmpl-{raw}-1a2b3c4g",
        f"chatcmpl-{raw}-1a2b3c4d-extra",
    ):
        assert not module.matches_scheduler_request_id(lookalike, request_ids)


def test_metric_total_sums_labeled_series() -> None:
    module = load_module()
    text = """
vllm:num_preemptions_total{engine="0"} 2
vllm:num_preemptions_total{engine="1"} 3
vllm:num_requests_running{engine="0"} 1
"""
    assert module.metric_total(text, "num_preemptions_total") == 5
    assert module.metric_total(text, "missing") is None
    assert module.metric_total("vllm:x NaN", "x") is None


def test_memory_monitor_samples_before_ready(monkeypatch) -> None:
    module = load_module()
    monkeypatch.setattr(module, "read_memfree_kib", lambda: 4 * 1024 * 1024)
    monkeypatch.setattr(
        module, "read_worker_memfree_kib", lambda _worker: 4 * 1024 * 1024
    )
    monitor = module.MemoryMonitor(
        "worker", tripwire_kib=3 * 1024 * 1024, interval=60
    )
    monitor.start(timeout=1)
    monitor.stop()
    assert monitor.ready.is_set()
    assert len(monitor.samples) == 1
    assert not monitor.breached.is_set()


def test_memory_monitor_publishes_initial_breach_before_ready(monkeypatch) -> None:
    module = load_module()
    monkeypatch.setattr(module, "read_memfree_kib", lambda: 2 * 1024 * 1024)
    monkeypatch.setattr(
        module, "read_worker_memfree_kib", lambda _worker: 4 * 1024 * 1024
    )
    monitor = module.MemoryMonitor(
        "worker", tripwire_kib=3 * 1024 * 1024, interval=60
    )
    monitor.start(timeout=1)
    monitor.stop()
    assert monitor.ready.is_set()
    assert monitor.breached.is_set()


def test_memory_monitor_publishes_initial_error_before_ready(monkeypatch) -> None:
    module = load_module()
    monkeypatch.setattr(
        module,
        "read_memfree_kib",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monitor = module.MemoryMonitor(
        "worker", tripwire_kib=3 * 1024 * 1024, interval=60
    )
    monitor.start(timeout=1)
    monitor.stop()
    assert monitor.ready.is_set()
    assert monitor.breached.is_set()
    assert monitor.error == "RuntimeError: boom"


def test_memory_monitor_stop_waits_for_outstanding_sample(monkeypatch) -> None:
    module = load_module()
    release = threading.Event()
    monkeypatch.setattr(module, "read_memfree_kib", lambda: 4 * 1024 * 1024)

    def delayed_worker(_worker):
        release.wait(1)
        return 2 * 1024 * 1024

    monkeypatch.setattr(module, "read_worker_memfree_kib", delayed_worker)
    monitor = module.MemoryMonitor(
        "worker", tripwire_kib=3 * 1024 * 1024, interval=60
    )
    monitor._thread.start()
    stopper = threading.Thread(target=monitor.stop)
    stopper.start()
    assert stopper.is_alive()
    release.set()
    stopper.join(1)
    assert not stopper.is_alive()
    assert monitor.breached.is_set()


def test_tripwire_cannot_be_lowered() -> None:
    module = load_module()
    assert module.tripwire_kib(str(3 * 1024 * 1024)) == 3 * 1024 * 1024
    for value in ("0", "-1", str(3 * 1024 * 1024 - 1)):
        try:
            module.tripwire_kib(value)
        except Exception as exc:
            assert "at least" in str(exc)
        else:
            raise AssertionError(f"accepted unsafe tripwire {value}")


def test_benchmark_receipt_requires_all_newcomer_ids(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "bench.json"
    path.write_text(
        json.dumps(
            {
                "samples": [
                    {"newcomer_request_id": "a"},
                    {"newcomer_request_id": "b"},
                ]
            }
        )
    )
    assert module.benchmark_request_ids(path) == {"a", "b"}
    path.write_text(json.dumps({"samples": [{"newcomer_request_id": "a"}, {}]}))
    try:
        module.benchmark_request_ids(path)
    except ValueError as exc:
        assert "request IDs" in str(exc)
    else:
        raise AssertionError("accepted missing request ID")


def test_runtime_logs_come_from_the_live_container(monkeypatch) -> None:
    module = load_module()
    calls = []
    result = SimpleNamespace(stdout="out", stderr="err")

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return result

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.read_runtime_logs("head", 1234) == "outerr"
    assert calls == [
        (
            ["docker", "logs", "--since", "1234", "head"],
            {
                "check": True,
                "capture_output": True,
                "text": True,
                "timeout": 60,
            },
        )
    ]


def test_signal_handlers_mark_interrupt(monkeypatch) -> None:
    module = load_module()
    installed = {}
    monkeypatch.setattr(
        module.signal,
        "signal",
        lambda signum, handler: installed.setdefault(signum, handler),
    )
    interrupted = threading.Event()
    module.install_signal_handlers(interrupted)
    for signum in (signal.SIGINT, signal.SIGTERM):
        interrupted.clear()
        installed[signum](signum, None)
        assert interrupted.is_set()


def test_run_benchmark_pins_endpoint_and_terminates_on_interrupt(
    monkeypatch, tmp_path: Path
) -> None:
    module = load_module()
    interrupted = threading.Event()
    interrupted.set()
    calls = {}

    class FakeProcess:
        def __init__(self, argv, env):
            calls["argv"] = argv
            calls["env"] = env
            self.terminated = False

        def poll(self):
            return None if not self.terminated else -15

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return -15

        def kill(self):
            self.terminated = True

    monkeypatch.setenv("GLM53_BASE", "http://wrong.example:9999")
    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    args = SimpleNamespace(samples=1, incumbent_tokens=8, timeout=10)
    monitor = SimpleNamespace(breached=threading.Event())
    rc = module.run_benchmark(
        args, tmp_path / "bench.json", monitor, interrupted
    )
    assert rc == -15
    assert calls["env"]["GLM53_BASE"] == module.BASE
