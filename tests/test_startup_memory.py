from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[1] / "experiments/startup-memory/wait_cluster_idle.py"
    spec = importlib.util.spec_from_file_location("startup_memory_gate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def idle(gate):
    return gate.Observation(gate.MIN_FREE_KIB, 110 * 1024 * 1024, gate.RESERVE_KIB, 0)


@pytest.mark.parametrize("fault", ["low_free", "container", "reserve", "error", "missing"])
def test_either_node_blocks_readiness(gate, fault):
    good = idle(gate)
    bad = {
        "low_free": replace(good, mem_free_kib=gate.MIN_FREE_KIB - 1),
        "container": replace(good, container_count=1),
        "reserve": replace(good, reserve_kib=2 * gate.RESERVE_KIB),
        "error": replace(good, error="SSH unavailable"),
        "missing": replace(good, mem_free_kib=None),
    }[fault]
    assert gate.all_ready({"head": good, "worker": good})
    assert not gate.all_ready({"head": good, "worker": bad})
    assert not gate.all_ready({"head": bad, "worker": good})
    assert not gate.all_ready({"head": good})


@pytest.mark.parametrize("fault", ["missing", "duplicate", "extra", "negative", "text"])
def test_malformed_probe_cannot_pass(gate, fault):
    text = "mem_free_kib=100000000\nmem_available_kib=110000000\nreserve_kib=1048576\ncontainer_count=0\n"
    assert gate.parse_probe(text).mem_free_kib == 100000000
    if fault == "missing":
        text = text.replace("container_count=0\n", "")
    elif fault == "duplicate":
        text += "mem_free_kib=100000000\n"
    elif fault == "extra":
        text += "ready=1\n"
    elif fault == "negative":
        text = text.replace("container_count=0", "container_count=-1")
    else:
        text = text.replace("100000000", "not-a-number")
    with pytest.raises(ValueError):
        gate.parse_probe(text)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


def test_does_not_reuse_old_head_success_when_worker_recovers(gate):
    clock = Clock()
    good = idle(gate)
    low = replace(good, mem_free_kib=1)
    samples = iter([good, low, low, good, good, good])
    calls = []

    def probe(node, worker, timeout):
        calls.append(node)
        return next(samples)

    result = gate.wait_for_cluster("worker", 30, 5, probe=probe, clock=clock, sleep=clock.sleep)
    assert result["ready"]
    assert [attempt["ready"] for attempt in result["attempts"]] == [False, False, True]
    assert calls == ["head", "worker"] * 3


def test_deadline_does_not_start_another_probe_or_accept_late_success(gate):
    clock = Clock()
    calls = []

    def probe(node, worker, timeout):
        calls.append((node, timeout))
        clock.now += timeout
        return idle(gate)

    result = gate.wait_for_cluster("worker", 3, 5, probe=probe, clock=clock, sleep=clock.sleep)
    assert not result["ready"]
    assert calls == [("head", 3)]
    assert result["attempts"][0]["nodes"]["worker"]["error"] == "readiness deadline reached"


def test_once_reports_both_nodes_without_waiting(gate):
    calls = []

    def probe(node, worker, timeout):
        calls.append(node)
        return replace(idle(gate), container_count=1)

    def no_sleep(_):
        pytest.fail("single-cycle probe must not sleep")

    result = gate.wait_for_cluster("worker", 30, 5, once=True, probe=probe, sleep=no_sleep)
    assert not result["ready"] and len(result["attempts"]) == 1
    assert calls == ["head", "worker"]


@pytest.mark.parametrize("fault", ["exit", "timeout", "missing_command", "malformed"])
def test_probe_failures_are_closed(gate, monkeypatch, fault):
    def run(command, **kwargs):
        if fault == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if fault == "missing_command":
            raise FileNotFoundError("ssh")
        if fault == "exit":
            return subprocess.CompletedProcess(command, 1, "", "docker unavailable")
        return subprocess.CompletedProcess(command, 0, "invalid", "")

    monkeypatch.setattr(gate.subprocess, "run", run)
    observation = gate.probe_node("worker", "nvidia@192.168.177.11", 5)
    assert observation.error
    assert gate.reason(observation)


def test_worker_probe_is_direct_bounded_and_includes_stopped_containers(gate, monkeypatch):
    observed = []

    def run(command, **kwargs):
        observed.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0,
            "mem_free_kib=100000000\nmem_available_kib=110000000\nreserve_kib=1048576\ncontainer_count=0\n", "")

    monkeypatch.setattr(gate.subprocess, "run", run)
    assert not gate.reason(gate.probe_node("worker", "nvidia@192.168.177.11", 5))
    command, kwargs = observed[0]
    assert command[0] == "ssh"
    assert "ProxyCommand=none" in command and "ProxyJump=none" in command
    assert command[-5:] == ["/usr/bin/timeout", "--kill-after=1", "8", "/bin/sh", "-s"]
    assert kwargs["timeout"] == 5
    assert "docker ps -aq" in kwargs["input"]
    assert "MemFree:" in kwargs["input"]


@pytest.mark.parametrize("value", ["-oProxyCommand=bad", "host; command", "host with spaces", "$(command)"])
def test_worker_address_rejects_shell_and_option_injection(gate, value):
    with pytest.raises(gate.argparse.ArgumentTypeError):
        gate.worker_address(value)
