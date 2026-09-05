#!/usr/bin/env python3
"""Host regression tests for the null-gap Mamba retirement overlay."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "overlay" / "patch_mamba_null_gap_retirement.py"

FIXTURE = """\
class KVCacheBlock:
    pass


class SingleTypeKVCacheManager:
    def _remove_blocks_in_range(
        self,
        request_id: str,
        first_block: int,
        last_block: int,
    ) -> None:
        blocks = self.req_to_blocks[request_id]
        freed: list[KVCacheBlock] = []
        for i in range(last_block - 1, first_block - 1, -1):
            if blocks[i] == self._null_block:
                break
            freed.append(blocks[i])
            blocks[i] = self._null_block
        if freed:
            self.block_pool.free_blocks(freed)


class MambaManager(SingleTypeKVCacheManager):
    def __init__(self):
        if self.mamba_cache_mode == "align":
            # Mapping from request ID to the index of the block
            # allocated in the previous step
            self.last_state_block_idx: dict[str, int] = {}
            # The set of the requests that have been allocated blocks
            self._allocated_block_reqs: set[str] = set()
            self._producer_partial_tail_reqs: dict[str, int] = {}

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        pass

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        if self.mamba_cache_mode == "align":
            self._allocated_block_reqs.discard(request_id)
            self.last_state_block_idx.pop(request_id, None)
            self._producer_partial_tail_reqs.pop(request_id, None)
        return []
"""


class Block:
    def __init__(self, block_id: int, *, is_null: bool = False) -> None:
        self.block_id = block_id
        self.is_null = is_null
        self.ref_cnt = 0 if is_null else 1


class Pool:
    def __init__(self) -> None:
        self.freed: list[Block] = []

    def free_blocks(self, blocks: list[Block]) -> None:
        self.freed.extend(blocks)
        for block in blocks:
            block.ref_cnt -= 1


def apply_patch(tmp_path: Path) -> Path:
    target = tmp_path / "single_type_kv_cache_manager.py"
    target.write_text(FIXTURE)
    env = os.environ.copy()
    env["GLM53_SINGLE_TYPE_KV_MANAGER_PY"] = str(target)
    subprocess.run([sys.executable, "-S", str(PATCH)], check=True, env=env)
    subprocess.run([sys.executable, "-S", str(PATCH)], check=True, env=env)
    return target


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("patched_mamba_manager", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overlay_retires_states_across_null_gap(tmp_path: Path) -> None:
    module = load_module(apply_patch(tmp_path))
    manager = module.MambaManager.__new__(module.MambaManager)
    manager.mamba_cache_mode = "align"
    manager._null_block = Block(-1, is_null=True)
    manager._num_retired_blocks = {}
    manager.block_pool = Pool()

    old = Block(0)
    committed = Block(1)
    in_flight = Block(2)
    manager.req_to_blocks = {
        "r": [old, manager._null_block, committed, in_flight]
    }

    manager._remove_blocks_in_range("r", 0, 2)
    assert manager.req_to_blocks["r"] == [
        manager._null_block,
        manager._null_block,
        committed,
        in_flight,
    ]
    assert old.ref_cnt == 0
    assert committed.ref_cnt == in_flight.ref_cnt == 1
    assert manager._num_retired_blocks["r"] == 2

    manager._remove_blocks_in_range("r", 0, 2)
    assert manager.block_pool.freed == [old]


def test_overlay_clears_request_retirement_state(tmp_path: Path) -> None:
    module = load_module(apply_patch(tmp_path))
    manager = module.MambaManager.__new__(module.MambaManager)
    manager.mamba_cache_mode = "align"
    manager._allocated_block_reqs = {"r"}
    manager.last_state_block_idx = {"r": 1}
    manager._num_retired_blocks = {"r": 2}
    manager._producer_partial_tail_reqs = {"r": 3}
    manager.pop_blocks_for_free("r")
    assert "r" not in manager._num_retired_blocks


def test_overlay_rejects_drifted_installed_body_without_writing(tmp_path: Path) -> None:
    target = apply_patch(tmp_path)
    drifted = target.read_text().replace(
        "if blocks[i].is_null:\n                continue",
        "if blocks[i].is_null:\n                break",
        1,
    )
    assert drifted != target.read_text()
    target.write_text(drifted)
    before = target.read_bytes()
    env = os.environ.copy()
    env["GLM53_SINGLE_TYPE_KV_MANAGER_PY"] = str(target)
    result = subprocess.run(
        [sys.executable, "-S", str(PATCH)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "expected exact Mamba retirement override" in result.stderr
    assert target.read_bytes() == before
