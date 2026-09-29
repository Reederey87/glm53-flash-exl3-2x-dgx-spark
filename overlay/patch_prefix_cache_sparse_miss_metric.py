#!/usr/bin/env python3
"""Expose shared-prefix tokens discarded during hybrid cache reconciliation.

This is a narrow, observability-only backport of vLLM PR #52527. The deployed
fork already computes the signal and stores its absolute endpoint in
``Request.shared_prefix_boundary``. This overlay:

* records ``max(shared_prefix_boundary - num_hits, 0)`` only when an admitted
  request's existing prefix-cache stats are recorded;
* carries the value through ``PrefixCacheStats``; and
* exports ``vllm:prefix_cache_sparse_retention_misses``.

It does not alter lookup, scheduling, retention, allocation, eviction, request
output, or model execution. In particular, it deliberately does not backport
PR #52527's later startup warning: this deployment already reports its
per-group retention posture, and its Mamba groups retain densely.

``GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC`` is an unset-to-enabled, exact 0/1
kill switch. A valid 0 leaves every target byte untouched. Any other value
fails closed before source files are inspected.

The installer is fail-closed, idempotent, all-files-preflighted, syntax
checked, mode preserving, and writes each target with an atomic rename.
"""

from __future__ import annotations

import ast
import os
import stat
import sys
import tempfile
from pathlib import Path

MARK = "# [glm53-prefix-cache-sparse-miss-metric]"
ENV_NAME = "GLM53_PREFIX_CACHE_SPARSE_MISS_METRIC"
METRIC_NAME = "vllm:prefix_cache_sparse_retention_misses"

_VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"
STATS_PY = Path(
    os.environ.get("GLM53_PREFIX_CACHE_STATS_PY", f"{_VLLM}/v1/metrics/stats.py")
)
KV_CACHE_MANAGER_PY = Path(
    os.environ.get(
        "GLM53_PREFIX_CACHE_MANAGER_PY", f"{_VLLM}/v1/core/kv_cache_manager.py"
    )
)
LOGGERS_PY = Path(
    os.environ.get(
        "GLM53_PREFIX_CACHE_LOGGERS_PY", f"{_VLLM}/v1/metrics/loggers.py"
    )
)
SCHEDULER_PY = Path(
    os.environ.get(
        "GLM53_PREFIX_CACHE_SCHEDULER_PY", f"{_VLLM}/v1/core/sched/scheduler.py"
    )
)


def enabled() -> bool:
    raw = os.environ.get(ENV_NAME, "1")
    if raw not in ("0", "1"):
        raise SystemExit(f"{ENV_NAME} must be exactly 0 or 1 (got {raw!r})")
    return raw == "1"


# vllm/v1/metrics/stats.py ---------------------------------------------------

STATS_OLD = '''    preempted_hits: int = 0
    """The `hits` number for preempted requests."""

    def record(self, num_tokens: int, num_hits: int, preempted: bool) -> None:
        """Aggregate request information into the stats."""
        if preempted:
'''

STATS_NEW = '''    preempted_hits: int = 0
    """The `hits` number for preempted requests."""

    sparse_retention_misses: int = 0  # [glm53-prefix-cache-sparse-miss-metric]
    """Tokens in a shared prefix matched by some KV cache group but not reused
    because another participating group held no checkpoint at that position.
    Counted for preempted and new requests alike."""

    def record(
        self,
        num_tokens: int,
        num_hits: int,
        preempted: bool,
        sparse_retention_misses: int = 0,
    ) -> None:
        """Aggregate request information into the stats."""
        self.sparse_retention_misses += sparse_retention_misses
        if preempted:
'''


# vllm/v1/core/kv_cache_manager.py ------------------------------------------

MANAGER_OLD = '''        assert self.prefix_cache_stats is not None
        self.prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=num_hits,
            preempted=request.num_preemptions > 0,
        )
'''

MANAGER_NEW = '''        assert self.prefix_cache_stats is not None
        # The stored junction is the reconciled hit plus the shared prefix
        # discarded by a lagging group. Record only the discarded suffix.
        sparse_retention_misses = (  # [glm53-prefix-cache-sparse-miss-metric]
            max(request.shared_prefix_boundary - num_hits, 0)
            if request.shared_prefix_boundary
            else 0
        )
        self.prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=num_hits,
            preempted=request.num_preemptions > 0,
            sparse_retention_misses=sparse_retention_misses,
        )
'''


# vllm/v1/metrics/loggers.py -------------------------------------------------

LOGGER_COUNTER_OLD = '''        self.counter_prefix_cache_hits = create_metric_per_engine(
            counter_prefix_cache_hits, per_engine_labelvalues
        )

        #
        # External - KV connector prefix cache
'''

LOGGER_COUNTER_NEW = '''        self.counter_prefix_cache_hits = create_metric_per_engine(
            counter_prefix_cache_hits, per_engine_labelvalues
        )

        counter_prefix_cache_sparse_retention_misses = self._counter_cls(
            name="vllm:prefix_cache_sparse_retention_misses",  # [glm53-prefix-cache-sparse-miss-metric]
            documentation=(
                "Shared-prefix tokens matched by some KV cache group but not "
                "reused because another participating group held no checkpoint "
                "at that position."
            ),
            labelnames=labelnames,
        )
        self.counter_prefix_cache_sparse_retention_misses = (
            create_metric_per_engine(
                counter_prefix_cache_sparse_retention_misses,
                per_engine_labelvalues,
            )
        )

        #
        # External - KV connector prefix cache
'''

LOGGER_RECORD_OLD = '''            self.counter_prefix_cache_hits[engine_idx].inc(
                scheduler_stats.prefix_cache_stats.hits
            )

            if scheduler_stats.connector_prefix_cache_stats is not None:
'''

LOGGER_RECORD_NEW = '''            self.counter_prefix_cache_hits[engine_idx].inc(
                scheduler_stats.prefix_cache_stats.hits
            )
            self.counter_prefix_cache_sparse_retention_misses[engine_idx].inc(
                scheduler_stats.prefix_cache_stats.sparse_retention_misses
            )  # [glm53-prefix-cache-sparse-miss-metric]

            if scheduler_stats.connector_prefix_cache_stats is not None:
'''


PLAN = {
    "stats.py": (
        STATS_PY,
        (("prefix-cache-stats", STATS_OLD, STATS_NEW),),
        ("@dataclass\nclass PrefixCacheStats(BaseCacheStats):\n",),
    ),
    "kv_cache_manager.py": (
        KV_CACHE_MANAGER_PY,
        (("admission-stats", MANAGER_OLD, MANAGER_NEW),),
        (
            "def record_prefix_cache_stats("
            "self, request: Request, num_hits: int) -> None:\n",
        ),
    ),
    "loggers.py": (
        LOGGERS_PY,
        (
            ("prometheus-counter", LOGGER_COUNTER_OLD, LOGGER_COUNTER_NEW),
            ("prometheus-record", LOGGER_RECORD_OLD, LOGGER_RECORD_NEW),
        ),
        (),
    ),
}

SCHEDULER_ADMISSION_CALL = '''                # Record at admission so unscheduled lookups are not counted.
                if did_prefix_cache_lookup:
                    self.kv_cache_manager.record_prefix_cache_stats(
                        request, num_new_local_computed_tokens
                    )
'''

SCHEDULER_ALLOCATION_FAILURE_EXIT = '''                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break
'''


def expected_marks(edits: tuple[tuple[str, str, str], ...]) -> int:
    return sum(new.count(MARK) - old.count(MARK) for _, old, new in edits)


def _parses(text: str, path: Path) -> None:
    try:
        ast.parse(text, str(path))
    except SyntaxError as exc:
        raise SystemExit(f"{path}: does not parse: {exc}") from None


def preflight(
    path: Path,
    edits: tuple[tuple[str, str, str], ...],
    requires: tuple[str, ...],
) -> str | None:
    """Return patched text, or ``None`` for a complete prior installation."""
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    text = path.read_text()
    have = text.count(MARK)
    want = expected_marks(edits)
    if have:
        missing = [label for label, _old, new in edits if text.count(new) != 1]
        if have != want or missing:
            raise SystemExit(
                f"{path}: carries {have} '{MARK}' marker(s) (complete = {want}) "
                f"and lacks verbatim snippet(s) {missing}; refusing partial state"
            )
        _parses(text, path)
        return None
    for needle in requires:
        if text.count(needle) != 1:
            raise SystemExit(f"{path}: prerequisite {needle.strip()!r} not unique")
    for label, old, new in edits:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{path}: expected one {label} target, found {count}")
        text = text.replace(old, new, 1)
    if text.count(MARK) != want:
        raise SystemExit(f"{path}: internal marker-count error")
    _parses(text, path)
    return text


def atomic_write(path: Path, text: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".glm53", dir=path.parent
    )
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main() -> int:
    if not enabled():
        print(f"{ENV_NAME}=0: sparse-miss metric overlay disabled")
        return 0

    if not SCHEDULER_PY.is_file():
        raise SystemExit(f"missing {SCHEDULER_PY}")
    scheduler_text = SCHEDULER_PY.read_text()
    _parses(scheduler_text, SCHEDULER_PY)
    for label, needle in (
        ("allocation-failure exit", SCHEDULER_ALLOCATION_FAILURE_EXIT),
        ("admission-time stats call", SCHEDULER_ADMISSION_CALL),
    ):
        count = scheduler_text.count(needle)
        if count != 1:
            raise SystemExit(
                f"{SCHEDULER_PY}: expected one {label} anchor, found {count}"
            )
    if scheduler_text.index(
        SCHEDULER_ALLOCATION_FAILURE_EXIT
    ) > scheduler_text.index(SCHEDULER_ADMISSION_CALL):
        raise SystemExit(
            f"{SCHEDULER_PY}: prefix-cache stats are not after allocation success"
        )

    staged: dict[str, tuple[Path, str]] = {}
    for name, (path, edits, requires) in PLAN.items():
        patched = preflight(path, edits, requires)
        if patched is None:
            print(f"{path.name}: {MARK} already present - skipping")
        else:
            staged[name] = (path, patched)

    # Verify every target before the first write.
    for name, (path, patched) in staged.items():
        atomic_write(path, patched)
        print(f"patched {path.name} ({name}: {expected_marks(PLAN[name][1])} marks)")

    if staged:
        print(f"enabled Prometheus counter {METRIC_NAME}")
    else:
        print("sparse-miss metric overlay already applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
