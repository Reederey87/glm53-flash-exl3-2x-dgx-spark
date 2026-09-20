#!/usr/bin/env python3
"""Regression tests for the `[glm53-apc-tail-floor]` overlay (task 45, site 1).

The overlay moves three prefix-cache tail registrations from `n` to `(n - 1)`
floored to the hash unit, so the deepest reusable state lands on a position the
cache finder can actually request. `KVCacheManager.get_computed_blocks` caps the
lookup at `num_tokens - 1`, so a registration at `n` is unreachable by
construction.

These tests drive the *patched code*, not a re-implementation. The synthetic
fixtures below carry the three anchor blocks verbatim from the deployed tree
(`vllm/v1/core/sched/scheduler.py` and
`vllm/v1/core/single_type_kv_cache_manager.py`); if those anchors drift, the
overlay fails closed on the cluster and these fixtures are what pins the
expected shape.

The behavioural contract, in one line: for a prompt of `n` tokens the registered
tail must never exceed the reachable ceiling `(n - 1) // unit * unit`, and it
must equal that ceiling whenever `n` lands on the hash grid.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "overlay" / "patch_apc_tail_boundary.py"

HASH_UNIT = 64
PAGE = 3584
SCHED_REL = "v1/core/sched/scheduler.py"
MGR_REL = "v1/core/single_type_kv_cache_manager.py"

# Verbatim excerpt: the scheduler half, anchor included unchanged.
SCHEDULER_SRC = '''
class Request:
    def __init__(self, num_computed_tokens, num_prompt_tokens, num_tokens,
                 shared_prefix_boundary=0):
        self.num_computed_tokens = num_computed_tokens
        self.num_prompt_tokens = num_prompt_tokens
        self.num_tokens = num_tokens
        self.shared_prefix_boundary = shared_prefix_boundary


class Scheduler:
    def __init__(self, block_size, max_num_scheduled_tokens,
                 long_prefill_token_threshold, hash_block_size=64,
                 use_eagle=False, mamba_partial_cache_hit=False):
        self.cache_config = type("C", (), {"block_size": block_size})()
        self.scheduler_config = type(
            "S", (), {"long_prefill_token_threshold": long_prefill_token_threshold}
        )()
        self.max_num_scheduled_tokens = max_num_scheduled_tokens
        self.hash_block_size = hash_block_size
        self.use_eagle = use_eagle
        self.mamba_partial_cache_hit = mamba_partial_cache_hit

    def _mamba_block_aligned_split(self, request, num_new_tokens,
                                   num_new_local_computed_tokens=0,
                                   num_external_computed_tokens=0):
        start = (request.num_computed_tokens
                 + num_new_local_computed_tokens
                 + num_external_computed_tokens)
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

# Verbatim excerpts: both manager halves, anchors included unchanged.
MANAGERS_SRC = '''
class BlockPool:
    def __init__(self, hash_block_size=64):
        self.hash_block_size = hash_block_size
        self.registered = []

    def cache_partial_block(self, request, block, num_tokens,
                            kv_cache_group_id, block_size):
        self.registered.append(num_tokens)
        return ("partial", num_tokens)


class _Req:
    def __init__(self, request_id, num_prompt_tokens):
        self.request_id = request_id
        self.num_prompt_tokens = num_prompt_tokens


class FullAttentionManager:
    supports_fine_grained_hash_lookup = True

    def __init__(self, block_pool, block_size=3584, kv_cache_group_id=0):
        self.block_pool = block_pool
        self.block_size = block_size
        self.kv_cache_group_id = kv_cache_group_id
        self.req_to_blocks = {}
        self._partial_hit_reqs = {}

    def _cache_partial_tail_block(self, request, num_tokens):
        hash_block_size = self.block_pool.hash_block_size
        boundary_tokens = request.num_prompt_tokens // hash_block_size * hash_block_size
        if boundary_tokens == 0 or boundary_tokens > num_tokens:
            return
        if boundary_tokens % self.block_size == 0:
            return
        blocks = self.req_to_blocks[request.request_id]
        block_idx = boundary_tokens // self.block_size
        if block_idx >= len(blocks):
            return
        self.block_pool.cache_partial_block(
            request=request,
            block=blocks[block_idx],
            num_tokens=boundary_tokens,
            kv_cache_group_id=self.kv_cache_group_id,
            block_size=self.block_size,
        )


class MambaManager:
    supports_fine_grained_hash_lookup = True

    def __init__(self, block_pool, block_size=3584, kv_cache_group_id=2):
        self.block_pool = block_pool
        self.block_size = block_size
        self.kv_cache_group_id = kv_cache_group_id
        self.req_to_blocks = {}
        self.num_cached_block = {}
        self._partial_hit_reqs = {}
        self._producer_partial_tail_reqs = {}

    def _cache_partial_tail_block(self, request, num_tokens):
        hash_block_size = self.block_pool.hash_block_size
        if self.block_size == hash_block_size:
            return None
        if num_tokens % self.block_size == 0:
            return None
        if num_tokens % hash_block_size != 0:
            return None
        latest_prompt_hash_boundary = (
            request.num_prompt_tokens // hash_block_size
        ) * hash_block_size
        if num_tokens != latest_prompt_hash_boundary:
            return None
        block_idx = num_tokens // self.block_size
        blocks = self.req_to_blocks[request.request_id]
        if block_idx >= len(blocks):
            return None
        source_block = blocks[block_idx]
        return self.block_pool.cache_partial_block(
            request=request,
            block=source_block,
            num_tokens=num_tokens,
            kv_cache_group_id=self.kv_cache_group_id,
            block_size=self.block_size,
        )
'''


def build_tree(tmp_path: Path) -> Path:
    root = tmp_path / "site"
    (root / "v1/core/sched").mkdir(parents=True)
    (root / "v1/core").mkdir(parents=True, exist_ok=True)
    (root / SCHED_REL).write_text(SCHEDULER_SRC)
    (root / MGR_REL).write_text(MANAGERS_SRC)
    return root


def apply_overlay(root: Path, *, armed: bool) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["GLM53_VLLM_SITE"] = str(root)
    env["GLM53_APC_TAIL_FLOOR"] = "1" if armed else "0"
    return subprocess.run(
        [sys.executable, str(OVERLAY)], env=env, capture_output=True, text=True
    )


def armed_tree(tmp_path: Path) -> Path:
    root = build_tree(tmp_path)
    first = apply_overlay(root, armed=True)
    assert first.returncode == 0, first.stderr
    # Idempotent: a second run must be a byte-identical no-op.
    before = (root / SCHED_REL).read_text(), (root / MGR_REL).read_text()
    second = apply_overlay(root, armed=True)
    assert second.returncode == 0, second.stderr
    assert "already patched" in second.stdout
    assert ((root / SCHED_REL).read_text(), (root / MGR_REL).read_text()) == before
    assert not list(root.rglob("*.tmp"))
    return root


def load(path: Path) -> dict:
    ns: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), ns)
    return ns


def ceiling(n: int, unit: int = HASH_UNIT) -> int:
    """Deepest position the cache finder may hand back: `num_tokens - 1` floored."""
    return (n - 1) // unit * unit


def chunk_ends(root: Path, n: int) -> list[int]:
    """Walk the prefill the way the scheduler does and return each chunk end.

    One page-sized scheduling budget per step, honouring the clip the splitter
    returns and clamping the final step to the prompt, exactly as
    ``allocate_slots`` does.
    """
    ns = load(root / SCHED_REL)
    sched = ns["Scheduler"](
        block_size=PAGE,
        max_num_scheduled_tokens=PAGE,
        long_prefill_token_threshold=0,
        hash_block_size=HASH_UNIT,
        use_eagle=True,
        mamba_partial_cache_hit=True,
    )
    req = ns["Request"](num_computed_tokens=0, num_prompt_tokens=n, num_tokens=n)
    ends: list[int] = []
    start = 0
    while start < n:
        req.num_computed_tokens = start
        got = sched._mamba_block_aligned_split(req, PAGE)
        assert got > 0, (n, start)
        start = min(start + got, n)
        ends.append(start)
        assert len(ends) < 10_000
    return ends


def register_tail(root: Path, n: int, manager: str) -> int | None:
    """Deepest tail position the manager registers across the prompt's chunks.

    ``cache_blocks`` runs once per scheduled step, so the manager sees every
    chunk end; a tail is registered only where a chunk ended exactly on the
    boundary the manager believes is the prompt's last one.
    """
    mns = load(root / MGR_REL)
    pool = mns["BlockPool"](hash_block_size=HASH_UNIT)
    mgr = mns[manager](block_pool=pool, block_size=PAGE)
    mreq = mns["_Req"]("r", n)
    mgr.req_to_blocks["r"] = [object()] * 64
    for end in chunk_ends(root, n):
        mgr._cache_partial_tail_block(mreq, end)
    return pool.registered[-1] if pool.registered else None


# --------------------------------------------------------------------------
# patch mechanics
# --------------------------------------------------------------------------


def test_unarmed_is_byte_neutral(tmp_path: Path) -> None:
    root = build_tree(tmp_path)
    before = (root / SCHED_REL).read_text(), (root / MGR_REL).read_text()
    r = apply_overlay(root, armed=False)
    assert r.returncode == 0, r.stderr
    assert "skipping" in r.stdout
    assert ((root / SCHED_REL).read_text(), (root / MGR_REL).read_text()) == before
    assert "[glm53-apc-tail-floor]" not in (root / SCHED_REL).read_text()


def test_armed_patches_all_three_sites_and_parses(tmp_path: Path) -> None:
    root = armed_tree(tmp_path)
    sched = (root / SCHED_REL).read_text()
    mgr = (root / MGR_REL).read_text()
    assert sched.count("[glm53-apc-tail-floor]") == 1
    assert mgr.count("[glm53-apc-tail-floor]") == 2
    compile(sched, SCHED_REL, "exec")
    compile(mgr, MGR_REL, "exec")
    # Every patched site must floor from (n - 1).
    assert "(request.num_prompt_tokens - 1) // self.hash_block_size" in sched
    assert "(request.num_prompt_tokens - 1) // hash_block_size * hash_block_size" in mgr
    assert "(request.num_prompt_tokens - 1) // hash_block_size\n        ) * hash_block_size" in mgr


def test_missing_target_fails_closed(tmp_path: Path) -> None:
    root = build_tree(tmp_path)
    (root / MGR_REL).unlink()
    r = apply_overlay(root, armed=True)
    assert r.returncode == 1
    assert "FATAL" in r.stderr


def test_anchor_drift_fails_closed_without_writing(tmp_path: Path) -> None:
    root = build_tree(tmp_path)
    p = root / SCHED_REL
    p.write_text(
        p.read_text().replace(
            "request.num_prompt_tokens // self.hash_block_size * self.hash_block_size",
            "request.num_prompt_tokens // 64 * 64",
        )
    )
    before = p.read_text()
    r = apply_overlay(root, armed=True)
    assert r.returncode != 0
    assert "expected exactly one anchor" in (r.stderr + r.stdout)
    assert p.read_text() == before
    assert not list(root.rglob("*.tmp"))


def test_marker_with_incomplete_edit_is_refused(tmp_path: Path) -> None:
    root = armed_tree(tmp_path)
    p = root / MGR_REL
    # Revert the mamba half while leaving the marker behind: a partial edit.
    text = p.read_text()
    text = text.replace(
        "latest_prompt_hash_boundary = (  # [glm53-apc-tail-floor]\n"
        "            (request.num_prompt_tokens - 1) // hash_block_size\n"
        "        ) * hash_block_size",
        "latest_prompt_hash_boundary = (\n"
        "            request.num_prompt_tokens // hash_block_size\n"
        "        ) * hash_block_size",
    )
    p.write_text(text)
    r = apply_overlay(root, armed=True)
    assert r.returncode != 0
    assert "incomplete" in (r.stderr + r.stdout)


# --------------------------------------------------------------------------
# behaviour: the registered tail must be reachable
# --------------------------------------------------------------------------

LADDER = (6464, 6500, 7168, 7360, 10000, 10752, 14336, 20000)


def test_registered_tail_is_reachable_for_every_ladder_length(tmp_path: Path) -> None:
    """The fix's whole contract: never register above the reachable ceiling."""
    root = armed_tree(tmp_path)
    for n in LADDER:
        for manager in ("FullAttentionManager", "MambaManager"):
            pos = register_tail(root, n, manager)
            assert pos is not None, (n, manager)
            assert pos <= ceiling(n), (n, manager, pos, ceiling(n))


def test_hash_aligned_prompts_now_reach_their_ceiling(tmp_path: Path) -> None:
    """Regression for the measured defect: hash-grid lengths fell a page short."""
    root = armed_tree(tmp_path)
    for n in (6464, 7168, 7360, 10752, 14336):
        assert n % HASH_UNIT == 0
        for manager in ("FullAttentionManager", "MambaManager"):
            pos = register_tail(root, n, manager)
            assert pos == ceiling(n), (n, manager, pos, ceiling(n))


def test_page_aligned_prompts_now_register_a_tail_at_all(tmp_path: Path) -> None:
    """Stock refuses a block-aligned tail outright, so no entry existed."""
    stock = build_tree(tmp_path / "stock")
    fixed = build_tree(tmp_path / "fixed")
    assert apply_overlay(fixed, armed=True).returncode == 0
    for n in (7168, 10752, 14336):
        assert n % PAGE == 0
        old_pos = register_tail(stock, n, "MambaManager")
        new_pos = register_tail(fixed, n, "MambaManager")
        assert old_pos is None, (n, old_pos)
        assert new_pos == ceiling(n), (n, new_pos)


def test_non_aligned_prompts_are_unchanged(tmp_path: Path) -> None:
    """The change must be invisible wherever the old arithmetic already worked."""
    stock = build_tree(tmp_path / "stock")
    fixed = build_tree(tmp_path / "fixed")
    assert apply_overlay(fixed, armed=True).returncode == 0
    for n in (6500, 10000, 20000, 6501, 9999, 3585):
        assert n % HASH_UNIT != 0
        for manager in ("FullAttentionManager", "MambaManager"):
            old_pos = register_tail(stock, n, manager)
            new_pos = register_tail(fixed, n, manager)
            assert old_pos == new_pos, (n, manager, old_pos, new_pos)
            assert new_pos is None or new_pos <= ceiling(n), (n, manager, new_pos)


def test_stock_registers_above_the_ceiling_on_the_hash_grid(tmp_path: Path) -> None:
    """Negative control: without the overlay the defect is visible."""
    stock = build_tree(tmp_path)
    offenders = []
    for n in (6464, 7360, 10000, 20000):
        for manager in ("FullAttentionManager", "MambaManager"):
            pos = register_tail(stock, n, manager)
            if pos is None or pos > ceiling(n):
                offenders.append((n, manager, pos))
    # 6464/7360 are hash-grid lengths: stock registers one unit too high.
    assert (6464, "MambaManager", 6464) in offenders
    assert (7360, "MambaManager", 7360) in offenders
    # 10000/20000 already agreed with the ceiling, so they are not offenders.
    assert all(n in (6464, 7360) for n, _, _ in offenders), offenders


def test_scheduler_clips_the_chunk_at_the_floored_tail(tmp_path: Path) -> None:
    """The extra stop is what materialises the state at the reachable position."""
    stock = build_tree(tmp_path / "stock")
    fixed = build_tree(tmp_path / "fixed")
    assert apply_overlay(fixed, armed=True).returncode == 0

    # n = 10752: stock prefill runs 3584 / 7168 / 10752. The fix inserts the
    # extra stop at 10688, so the state at the reachable position exists.
    assert chunk_ends(stock, 10752) == [3584, 7168, 10752]
    assert chunk_ends(fixed, 10752) == [3584, 7168, 10688, 10752]
    # n = 7168: the second chunk is clipped from 7168 to 7104.
    assert chunk_ends(stock, 7168) == [3584, 7168]
    assert chunk_ends(fixed, 7168) == [3584, 7104, 7168]
    # A non-aligned prompt is untouched by the extra stop.
    assert chunk_ends(stock, 10000) == chunk_ends(fixed, 10000) == [3584, 7168, 9984, 10000]


def test_page_aligned_stop_is_not_swallowed_by_last_cache_position(tmp_path: Path) -> None:
    """The stock guard `last_cache_position < tail_boundary` excluded the tail.

    On a page-aligned prompt `last_cache_position` backs off a whole page under
    EAGLE while the stock `tail_boundary` equals `num_prompt_tokens`, so the
    strict inequality failed and the stop was never emitted.
    """
    ns = load(armed_tree(tmp_path) / SCHED_REL)
    n = 10752
    sched = ns["Scheduler"](
        block_size=PAGE,
        max_num_scheduled_tokens=PAGE,
        long_prefill_token_threshold=0,
        hash_block_size=HASH_UNIT,
        use_eagle=True,
        mamba_partial_cache_hit=True,
    )
    last_cache_position = max(n - n % PAGE - PAGE, 0)
    assert last_cache_position == 7168
    assert last_cache_position < ceiling(n) < n
