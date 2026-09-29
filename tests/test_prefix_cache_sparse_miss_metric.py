#!/usr/bin/env python3
"""Host-only tests for the sparse-retention miss metric overlay."""

from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
PATCH = ROOT / "overlay" / "patch_prefix_cache_sparse_miss_metric.py"

STATS_FIXTURE = '''\
from dataclasses import dataclass

@dataclass
class BaseCacheStats:
    requests: int = 0
    queries: int = 0
    hits: int = 0

@dataclass
class PrefixCacheStats(BaseCacheStats):
    """
    Stores prefix cache hit statistics.
    - `reset`: Whether `reset_prefix_cache` was invoked.
    - `queries`: Refers to the number of tokens that were queried.
    """

    preempted_requests: int = 0
    """The number of previously preempted requests in this update."""

    preempted_queries: int = 0
    """The `queries` number for preempted requests."""

    preempted_hits: int = 0
    """The `hits` number for preempted requests."""

    def record(self, num_tokens: int, num_hits: int, preempted: bool) -> None:
        """Aggregate request information into the stats."""
        if preempted:
            # Previously preempted request
            self.preempted_requests += 1
            self.preempted_queries += num_tokens
            self.preempted_hits += num_hits
        else:
            # New request
            self.requests += 1
            self.queries += num_tokens
            self.hits += num_hits
'''

MANAGER_FIXTURE = '''\
class Request:
    pass

class KVCacheManager:
    def __init__(self, stats, *, log_stats=True, cache_enabled=True):
        self.prefix_cache_stats = stats
        self.log_stats = log_stats
        self.cache_enabled = cache_enabled
        self.allocation_results = []

    def allocate_slots(self):
        return self.allocation_results.pop(0)

    def make_prefix_cache_stats(self):
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def prefix_cache_lookup_enabled(self, request):
        return self.cache_enabled and not request.skip_reading_prefix_cache

    def record_prefix_cache_stats(self, request: Request, num_hits: int) -> None:
        # Don't count a request that skipped the cache lookup.
        if not self.log_stats or not self.prefix_cache_lookup_enabled(request):
            return
        assert self.prefix_cache_stats is not None
        self.prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=num_hits,
            preempted=request.num_preemptions > 0,
        )
'''

LOGGERS_FIXTURE = '''\
class PrometheusStatLogger:
    def __init__(self, labelnames, per_engine_labelvalues):
        counter_prefix_cache_queries = self._counter_cls(
            name="vllm:prefix_cache_queries",
            documentation=(
                "Prefix cache queries, in terms of number of queried tokens."
            ),
            labelnames=labelnames,
        )
        self.counter_prefix_cache_queries = create_metric_per_engine(
            counter_prefix_cache_queries, per_engine_labelvalues
        )

        counter_prefix_cache_hits = self._counter_cls(
            name="vllm:prefix_cache_hits",
            documentation=("Prefix cache hits, in terms of number of cached tokens."),
            labelnames=labelnames,
        )
        self.counter_prefix_cache_hits = create_metric_per_engine(
            counter_prefix_cache_hits, per_engine_labelvalues
        )

        #
        # External - KV connector prefix cache
        #
        self.connector_placeholder = True

    def record(self, scheduler_stats, engine_idx=0):
        if scheduler_stats is not None:
            self.counter_prefix_cache_queries[engine_idx].inc(
                scheduler_stats.prefix_cache_stats.queries
            )
            self.counter_prefix_cache_hits[engine_idx].inc(
                scheduler_stats.prefix_cache_stats.hits
            )

            if scheduler_stats.connector_prefix_cache_stats is not None:
                pass
'''

SCHEDULER_FIXTURE = '''\
class Scheduler:
    def __init__(self, kv_cache_manager):
        self.kv_cache_manager = kv_cache_manager
        self.encoder_cache_manager = None

    def schedule(
        self, request, num_new_local_computed_tokens,
        did_prefix_cache_lookup=True,
    ):
        while True:
            if request:
                new_blocks = self.kv_cache_manager.allocate_slots()
                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # Record at admission so unscheduled lookups are not counted.
                if did_prefix_cache_lookup:
                    self.kv_cache_manager.record_prefix_cache_stats(
                        request, num_new_local_computed_tokens
                    )
                return True
        return False
'''


def _load_patcher(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    paths = {
        "GLM53_PREFIX_CACHE_STATS_PY": tmp_path / "stats.py",
        "GLM53_PREFIX_CACHE_MANAGER_PY": tmp_path / "kv_cache_manager.py",
        "GLM53_PREFIX_CACHE_LOGGERS_PY": tmp_path / "loggers.py",
        "GLM53_PREFIX_CACHE_SCHEDULER_PY": tmp_path / "scheduler.py",
    }
    for name, path in paths.items():
        monkeypatch.setenv(name, str(path))
    spec = importlib.util.spec_from_file_location(
        f"patch_sparse_miss_{id(tmp_path)}", PATCH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, paths


def _write_fixtures(paths: dict[str, Path]) -> None:
    paths["GLM53_PREFIX_CACHE_STATS_PY"].write_text(STATS_FIXTURE)
    paths["GLM53_PREFIX_CACHE_MANAGER_PY"].write_text(MANAGER_FIXTURE)
    paths["GLM53_PREFIX_CACHE_LOGGERS_PY"].write_text(LOGGERS_FIXTURE)
    paths["GLM53_PREFIX_CACHE_SCHEDULER_PY"].write_text(SCHEDULER_FIXTURE)


def _run_patch(paths: dict[str, Path], value: str = "1"):
    env = os.environ.copy()
    env.update({name: str(path) for name, path in paths.items()})
    env["GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC"] = value
    return subprocess.run(
        [sys.executable, str(PATCH)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _load_classes(paths: dict[str, Path]):
    stats_ns: dict[str, object] = {}
    exec(compile(paths["GLM53_PREFIX_CACHE_STATS_PY"].read_text(), "stats.py", "exec"), stats_ns)
    manager_ns: dict[str, object] = {
        "PrefixCacheStats": stats_ns["PrefixCacheStats"],
    }
    exec(
        compile(
            paths["GLM53_PREFIX_CACHE_MANAGER_PY"].read_text(),
            "kv_cache_manager.py",
            "exec",
        ),
        manager_ns,
    )
    return stats_ns["PrefixCacheStats"], manager_ns["KVCacheManager"]


def _load_scheduler(path: Path):
    namespace: dict[str, object] = {}
    exec(compile(path.read_text(), path.name, "exec"), namespace)
    return namespace["Scheduler"]


def _request(*, tokens=64, boundary=0, preemptions=0, skip=False):
    return SimpleNamespace(
        num_tokens=tokens,
        shared_prefix_boundary=boundary,
        num_preemptions=preemptions,
        skip_reading_prefix_cache=skip,
    )


def test_installer_applies_idempotently_and_preserves_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patcher, paths = _load_patcher(monkeypatch, tmp_path)
    _write_fixtures(paths)
    for index, path in enumerate(paths.values()):
        path.chmod(0o640 + index)
    before_modes = {path: stat.S_IMODE(path.stat().st_mode) for path in paths.values()}

    first = _run_patch(paths)
    assert first.returncode == 0, first.stderr
    after_first = {path: path.read_bytes() for path in paths.values()}
    assert {path: stat.S_IMODE(path.stat().st_mode) for path in paths.values()} == before_modes
    for name, (path, edits, _requires) in patcher.PLAN.items():
        assert path.read_text().count(patcher.MARK) == patcher.expected_marks(edits), name
        compile(path.read_text(), path.name, "exec")

    second = _run_patch(paths)
    assert second.returncode == 0, second.stderr
    assert {path: path.read_bytes() for path in paths.values()} == after_first


@pytest.mark.parametrize("value", ["", "yes", "2", "-1"])
def test_invalid_switch_fails_without_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str
) -> None:
    _patcher, paths = _load_patcher(monkeypatch, tmp_path)
    _write_fixtures(paths)
    before = {path: path.read_bytes() for path in paths.values()}
    result = _run_patch(paths, value)
    assert result.returncode != 0
    assert {path: path.read_bytes() for path in paths.values()} == before


def test_disabled_switch_is_byte_exact_noop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patcher, paths = _load_patcher(monkeypatch, tmp_path)
    _write_fixtures(paths)
    before = {path: path.read_bytes() for path in paths.values()}
    result = _run_patch(paths, "0")
    assert result.returncode == 0
    assert "disabled" in result.stdout
    assert {path: path.read_bytes() for path in paths.values()} == before


def test_anchor_drift_is_transactional(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patcher, paths = _load_patcher(monkeypatch, tmp_path)
    _write_fixtures(paths)
    paths["GLM53_PREFIX_CACHE_LOGGERS_PY"].write_text(
        LOGGERS_FIXTURE.replace("External - KV connector", "External KV connector")
    )
    before = {path: path.read_bytes() for path in paths.values()}
    result = _run_patch(paths)
    assert result.returncode != 0
    assert {path: path.read_bytes() for path in paths.values()} == before


def test_allocation_failure_must_exit_before_stats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Replacing the deployed failure branch's break with pass is refused."""
    _patcher, paths = _load_patcher(monkeypatch, tmp_path)
    _write_fixtures(paths)
    scheduler = paths["GLM53_PREFIX_CACHE_SCHEDULER_PY"]
    scheduler.write_text(
        scheduler.read_text().replace(
            "                    break\n\n"
            "                # Record at admission",
            "                    pass\n\n"
            "                # Record at admission",
            1,
        )
    )
    before = {path: path.read_bytes() for path in paths.values()}
    result = _run_patch(paths)
    assert result.returncode != 0
    assert "allocation-failure exit" in result.stderr
    assert {path: path.read_bytes() for path in paths.values()} == before


def test_partial_marker_state_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patcher, paths = _load_patcher(monkeypatch, tmp_path)
    _write_fixtures(paths)
    stats = paths["GLM53_PREFIX_CACHE_STATS_PY"]
    stats.write_text(stats.read_text() + f"\n{patcher.MARK}\n")
    before = {path: path.read_bytes() for path in paths.values()}
    result = _run_patch(paths)
    assert result.returncode != 0
    assert {path: path.read_bytes() for path in paths.values()} == before


def test_positive_loss_healthy_zero_and_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patcher, paths = _load_patcher(monkeypatch, tmp_path)
    _write_fixtures(paths)
    assert _run_patch(paths).returncode == 0
    Stats, Manager = _load_classes(paths)
    stats = Stats()
    manager = Manager(stats)

    manager.record_prefix_cache_stats(_request(boundary=32), 0)
    manager.record_prefix_cache_stats(_request(boundary=64), 64)
    manager.record_prefix_cache_stats(_request(boundary=32), 48)
    assert stats.hits == 112
    assert stats.sparse_retention_misses == 32


def test_preemption_counts_loss_without_double_counting_new_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patcher, paths = _load_patcher(monkeypatch, tmp_path)
    _write_fixtures(paths)
    assert _run_patch(paths).returncode == 0
    Stats, Manager = _load_classes(paths)
    stats = Stats()
    manager = Manager(stats)
    manager.record_prefix_cache_stats(
        _request(tokens=80, boundary=48, preemptions=1), 16
    )
    assert stats.requests == 0
    assert stats.preempted_requests == 1
    assert stats.preempted_hits == 16
    assert stats.sparse_retention_misses == 32


@pytest.mark.parametrize(
    ("manager_kwargs", "request_kwargs"),
    [
        ({"log_stats": False}, {}),
        ({"cache_enabled": False}, {}),
        ({}, {"skip": True}),
    ],
)
def test_disabled_stats_or_lookup_records_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manager_kwargs: dict[str, bool],
    request_kwargs: dict[str, bool],
) -> None:
    _patcher, paths = _load_patcher(monkeypatch, tmp_path)
    _write_fixtures(paths)
    assert _run_patch(paths).returncode == 0
    Stats, Manager = _load_classes(paths)
    stats = Stats()
    Manager(stats, **manager_kwargs).record_prefix_cache_stats(
        _request(boundary=32, **request_kwargs), 0
    )
    assert stats.sparse_retention_misses == 0
    assert stats.requests == 0


def test_admission_deferral_retry_records_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed allocation emits nothing; the later successful admission does."""
    _patcher, paths = _load_patcher(monkeypatch, tmp_path)
    _write_fixtures(paths)
    assert _run_patch(paths).returncode == 0
    Stats, Manager = _load_classes(paths)
    stats = Stats()
    manager = Manager(stats)
    Scheduler = _load_scheduler(paths["GLM53_PREFIX_CACHE_SCHEDULER_PY"])
    scheduler = Scheduler(manager)
    request = _request(boundary=32)
    request.has_encoder_inputs = False
    manager.allocation_results = [None, object()]

    assert scheduler.schedule(request, 0) is False
    assert stats.sparse_retention_misses == 0
    assert scheduler.schedule(request, 0) is True
    assert stats.sparse_retention_misses == 32
    assert stats.requests == 1


class _Counter:
    def __init__(self, name: str):
        self.name = name
        self.value = 0

    def inc(self, value: int) -> None:
        self.value += value


def _load_logger(path: Path):
    created: dict[str, _Counter] = {}

    def counter_cls(*, name, documentation, labelnames):
        del documentation, labelnames
        counter = _Counter(name)
        created[name] = counter
        return counter

    def per_engine(counter, values):
        return {index: counter for index, _value in enumerate(values)}

    namespace = {
        "create_metric_per_engine": per_engine,
    }
    exec(compile(path.read_text(), path.name, "exec"), namespace)
    Logger = namespace["PrometheusStatLogger"]
    Logger._counter_cls = staticmethod(counter_cls)
    return Logger, created


def test_admission_to_prometheus_export_and_drain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patcher, paths = _load_patcher(monkeypatch, tmp_path)
    _write_fixtures(paths)
    assert _run_patch(paths).returncode == 0
    Stats, Manager = _load_classes(paths)
    stats = Stats()
    manager = Manager(stats)
    manager.record_prefix_cache_stats(_request(boundary=32), 0)

    Logger, created = _load_logger(paths["GLM53_PREFIX_CACHE_LOGGERS_PY"])
    logger = Logger(["model"], [{"model": "glm"}])
    scheduler_stats = SimpleNamespace(
        prefix_cache_stats=manager.make_prefix_cache_stats(),
        connector_prefix_cache_stats=None,
    )
    logger.record(scheduler_stats)
    assert (
        created["vllm:prefix_cache_sparse_retention_misses"].value == 32
    )

    # Exercise the deployed manager's actual drain-and-reset method. The
    # cumulative Prometheus counter must not increment again.
    scheduler_stats.prefix_cache_stats = manager.make_prefix_cache_stats()
    logger.record(scheduler_stats)
    assert (
        created["vllm:prefix_cache_sparse_retention_misses"].value == 32
    )


def test_negative_control_removing_drain_reset_double_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A mutant that keeps the old accumulator is detected by two exports."""
    _patcher, paths = _load_patcher(monkeypatch, tmp_path)
    _write_fixtures(paths)
    assert _run_patch(paths).returncode == 0
    manager_path = paths["GLM53_PREFIX_CACHE_MANAGER_PY"]
    manager_path.write_text(
        manager_path.read_text().replace(
            "        self.prefix_cache_stats = PrefixCacheStats()\n"
            "        return stats\n",
            "        self.prefix_cache_stats = stats  # negative-control\n"
            "        return stats\n",
            1,
        )
    )
    Stats, Manager = _load_classes(paths)
    manager = Manager(Stats())
    manager.record_prefix_cache_stats(_request(boundary=32), 0)
    Logger, created = _load_logger(paths["GLM53_PREFIX_CACHE_LOGGERS_PY"])
    logger = Logger(["model"], [{"model": "glm"}])
    scheduler_stats = SimpleNamespace(
        prefix_cache_stats=manager.make_prefix_cache_stats(),
        connector_prefix_cache_stats=None,
    )
    logger.record(scheduler_stats)
    scheduler_stats.prefix_cache_stats = manager.make_prefix_cache_stats()
    logger.record(scheduler_stats)
    assert (
        created["vllm:prefix_cache_sparse_retention_misses"].value != 32
    )


def test_negative_control_removing_boundary_signal_breaks_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patcher, paths = _load_patcher(monkeypatch, tmp_path)
    _write_fixtures(paths)
    assert _run_patch(paths).returncode == 0
    manager_path = paths["GLM53_PREFIX_CACHE_MANAGER_PY"]
    manager_path.write_text(
        manager_path.read_text().replace(
            "max(request.shared_prefix_boundary - num_hits, 0)",
            "0  # negative-control: signal removed",
            1,
        )
    )
    Stats, Manager = _load_classes(paths)
    stats = Stats()
    Manager(stats).record_prefix_cache_stats(_request(boundary=32), 0)
    assert stats.sparse_retention_misses != 32


def test_overlay_does_not_patch_scheduler_or_coordinator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patcher, _paths = _load_patcher(monkeypatch, tmp_path)
    targets = {path.name for path, _edits, _requires in patcher.PLAN.values()}
    assert targets == {"stats.py", "kv_cache_manager.py", "loggers.py"}


def test_launcher_wiring_contract() -> None:
    start = (ROOT / "start.sh").read_text()
    expected = [
        'PREFIX_CACHE_SPARSE_MISS_PATCH_HOST="${PREFIX_CACHE_SPARSE_MISS_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_prefix_cache_sparse_miss_metric.py}"',
        'GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC="${GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC-1}"',
        "GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC",
        "/opt/glm53/patch_prefix_cache_sparse_miss_metric.py",
        "/tmp/patch_prefix_cache_sparse_miss_metric.py",
    ]
    for needle in expected:
        assert needle in start
    assert start.count(
        "python3 -S /opt/glm53/patch_prefix_cache_sparse_miss_metric.py"
    ) == 2
    assert start.count(
        "/opt/glm53/patch_prefix_cache_sparse_miss_metric.py:ro"
    ) == 2
    assert (
        "GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC=1"
        in (ROOT / "env.example").read_text()
    )


def test_source_contains_no_startup_retention_warning() -> None:
    source = PATCH.read_text()
    assert "retain only semantic checkpoints" not in source
    assert "prefix_cache_retention_interval is 0" not in source


def test_real_installed_sources_when_present(tmp_path: Path) -> None:
    vllm = Path("/usr/local/lib/python3.12/dist-packages/vllm")
    source_paths = {
        "GLM53_PREFIX_CACHE_STATS_PY": vllm / "v1/metrics/stats.py",
        "GLM53_PREFIX_CACHE_MANAGER_PY": vllm / "v1/core/kv_cache_manager.py",
        "GLM53_PREFIX_CACHE_LOGGERS_PY": vllm / "v1/metrics/loggers.py",
        "GLM53_PREFIX_CACHE_SCHEDULER_PY": vllm / "v1/core/sched/scheduler.py",
    }
    if not all(path.is_file() for path in source_paths.values()):
        if os.environ.get("GLM53_REQUIRE_TARGET") == "1":
            pytest.fail("GLM53_REQUIRE_TARGET=1 but installed vLLM sources are absent")
        pytest.skip("installed vLLM sources are absent")

    copies: dict[str, Path] = {}
    for env_name, source in source_paths.items():
        target = tmp_path / source.name
        target.write_bytes(source.read_bytes())
        copies[env_name] = target
    result = _run_patch(copies)
    assert result.returncode == 0, result.stderr
    for path in copies.values():
        compile(path.read_text(), path.name, "exec")
