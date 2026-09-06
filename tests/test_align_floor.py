#!/usr/bin/env python3
"""Regression tests for the Mamba mixed-prefill align-floor overlay.

The production-equivalence sweep keeps the historical 608-case contract:
19 scheduler positions x 8 chunk sizes x 4 scheduling budgets at the deployed
LPTT=1792.  In that regime the overlay must be dormant, including not reading
its cached environment gate.
"""
from __future__ import annotations

import os
import subprocess
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "overlay" / "patch_align_floor.py"
BLOCK_SIZE = 3584

SYNTHETIC = '''import itertools
import time
from dataclasses import dataclass
from vllm.compilation.cuda_graph import CUDAGraphStat


@dataclass
class Request:
    num_computed_tokens: int
    num_prompt_tokens: int
    num_tokens: int
    shared_prefix_boundary: int = 0


class Scheduler:
    def __init__(
        self,
        *,
        block_size,
        max_num_scheduled_tokens,
        long_prefill_token_threshold,
        hash_block_size=64,
        use_eagle=False,
        mamba_partial_cache_hit=False,
    ):
        self.cache_config = type("CacheConfig", (), {"block_size": block_size})()
        self.scheduler_config = type(
            "SchedulerConfig",
            (),
            {"long_prefill_token_threshold": long_prefill_token_threshold},
        )()
        self.max_num_scheduled_tokens = max_num_scheduled_tokens
        self.hash_block_size = hash_block_size
        self.use_eagle = use_eagle
        self.mamba_partial_cache_hit = mamba_partial_cache_hit

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        """Clip a prefill chunk so it ends where Mamba state must be cached."""
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if start >= prefill_end:
            return num_new_tokens

        block_size = self.cache_config.block_size
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        if self.use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)

        end = start + num_new_tokens
        if end < prefill_end:
            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end

        next_block_boundary = (start // block_size + 1) * block_size
        tail_boundary = (
            request.num_prompt_tokens // self.hash_block_size * self.hash_block_size
            if self.mamba_partial_cache_hit
            else 0
        )
        stops = (
            next_block_boundary if start % block_size != 0 else 0,
            last_cache_position,
            tail_boundary
            if last_cache_position < tail_boundary < request.num_prompt_tokens
            else 0,
            start + (request.shared_prefix_boundary - start) // block_size * block_size
            if start < request.shared_prefix_boundary < end
            else 0,
        )
        end = min((s for s in stops if start < s < end), default=end)
        return max(end - start, 0)
'''

IMPORT_STUB = "from vllm.compilation.cuda_graph import CUDAGraphStat"


def patch_source(tmp_path: Path) -> str:
    target = tmp_path / "scheduler.py"
    target.write_text(SYNTHETIC)
    env = os.environ.copy()
    env["GLM53_SCHEDULER_PY"] = str(target)
    first = subprocess.run(
        [sys.executable, str(OVERLAY)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stderr
    patched = target.read_text()
    compile(patched, str(target), "exec")

    second = subprocess.run(
        [sys.executable, str(OVERLAY)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 0, second.stderr
    assert "already present and complete" in second.stdout
    assert target.read_text() == patched
    assert not list(tmp_path.glob("*.tmp"))
    return patched


def load_source(source: str) -> dict[str, object]:
    namespace: dict[str, object] = {}
    executable = source.replace(IMPORT_STUB, "CUDAGraphStat = object")
    exec(compile(executable, "scheduler.py", "exec"), namespace)
    return namespace


def run_split(
    source: str,
    *,
    align_floor: bool,
    start: int,
    chunk: int,
    lptt: int,
    max_scheduled: int = 8192,
    prompt_tokens: int = 20_000,
    num_tokens: int | None = None,
    shared_prefix_boundary: int = 0,
    use_eagle: bool = False,
    partial_hit: bool = False,
) -> tuple[int, object]:
    old = os.environ.get("GLM53_ALIGN_FLOOR")
    os.environ["GLM53_ALIGN_FLOOR"] = "1" if align_floor else "0"
    try:
        namespace = load_source(source)
        scheduler = namespace["Scheduler"](
            block_size=BLOCK_SIZE,
            max_num_scheduled_tokens=max_scheduled,
            long_prefill_token_threshold=lptt,
            use_eagle=use_eagle,
            mamba_partial_cache_hit=partial_hit,
        )
        request = namespace["Request"](
            num_computed_tokens=start,
            num_prompt_tokens=prompt_tokens,
            num_tokens=prompt_tokens if num_tokens is None else num_tokens,
            shared_prefix_boundary=shared_prefix_boundary,
        )
        result = scheduler._mamba_block_aligned_split(request, chunk)
        return result, namespace.get("_GLM53_ALIGN_FLOOR")
    finally:
        if old is None:
            os.environ.pop("GLM53_ALIGN_FLOOR", None)
        else:
            os.environ["GLM53_ALIGN_FLOOR"] = old


def test_deployed_lptt_is_dormant_over_608_scheduler_combinations(
    tmp_path: Path,
) -> None:
    patched = patch_source(tmp_path)
    starts = (
        0,
        1,
        63,
        64,
        511,
        512,
        1023,
        1024,
        1535,
        1536,
        1791,
        1792,
        2047,
        2048,
        3071,
        3072,
        3583,
        3584,
        7167,
    )
    chunks = (0, 1, 63, 64, 511, 512, 1024, 1792)
    scheduling_budgets = (512, 1024, 1792, 3584)
    cases = list(product(starts, chunks, scheduling_budgets))
    assert len(cases) == 608

    for start, chunk, max_scheduled in cases:
        enabled, enabled_cache = run_split(
            patched,
            align_floor=True,
            start=start,
            chunk=chunk,
            lptt=1792,
            max_scheduled=max_scheduled,
        )
        rollback, rollback_cache = run_split(
            patched,
            align_floor=False,
            start=start,
            chunk=chunk,
            lptt=1792,
            max_scheduled=max_scheduled,
        )
        assert enabled == rollback, (start, chunk, max_scheduled)
        assert enabled_cache is None
        assert rollback_cache is None


def test_lptt_boundary_activates_only_at_one_full_page(tmp_path: Path) -> None:
    patched = patch_source(tmp_path)

    below, below_cache = run_split(
        patched,
        align_floor=True,
        start=0,
        chunk=512,
        lptt=BLOCK_SIZE - 1,
    )
    at_boundary, at_cache = run_split(
        patched,
        align_floor=True,
        start=0,
        chunk=512,
        lptt=BLOCK_SIZE,
    )
    above, above_cache = run_split(
        patched,
        align_floor=True,
        start=0,
        chunk=512,
        lptt=BLOCK_SIZE + 1,
    )
    rollback, rollback_cache = run_split(
        patched,
        align_floor=False,
        start=0,
        chunk=512,
        lptt=BLOCK_SIZE,
    )

    assert below == 512
    assert below_cache is None
    assert (at_boundary, above) == (512, 512)
    assert (at_cache, above_cache) == (True, True)
    assert rollback == 0
    assert rollback_cache is False


def test_sub_block_decode_floor_caps_keep_forward_progress(tmp_path: Path) -> None:
    patched = patch_source(tmp_path)

    for chunk in (512, 1024, 1792):
        fixed, _ = run_split(
            patched,
            align_floor=True,
            start=0,
            chunk=chunk,
            lptt=BLOCK_SIZE,
        )
        upstream, _ = run_split(
            patched,
            align_floor=False,
            start=0,
            chunk=chunk,
            lptt=BLOCK_SIZE,
        )
        assert fixed == chunk
        assert upstream == 0


def test_mid_block_chunk_stops_at_next_page_boundary(tmp_path: Path) -> None:
    patched = patch_source(tmp_path)
    start = BLOCK_SIZE - 84

    fixed, fixed_cache = run_split(
        patched,
        align_floor=True,
        start=start,
        chunk=512,
        lptt=BLOCK_SIZE,
    )
    rollback, rollback_cache = run_split(
        patched,
        align_floor=False,
        start=start,
        chunk=512,
        lptt=BLOCK_SIZE,
    )

    assert fixed == rollback == 84
    assert fixed_cache is None
    assert rollback_cache is None


def test_zero_input_and_tail_stop_are_unchanged(tmp_path: Path) -> None:
    patched = patch_source(tmp_path)

    for enabled in (False, True):
        zero, _ = run_split(
            patched,
            align_floor=enabled,
            start=1024,
            chunk=0,
            lptt=BLOCK_SIZE,
        )
        tail, _ = run_split(
            patched,
            align_floor=enabled,
            start=BLOCK_SIZE,
            chunk=512,
            lptt=BLOCK_SIZE,
            prompt_tokens=3700,
            partial_hit=True,
        )
        assert zero == 0
        assert tail == 64
