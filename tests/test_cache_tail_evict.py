"""Tail-first prefix eviction: ranking, and the installer around it."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "overlay"))

from cache_tail_evict import rank_key, select_blocks  # noqa: E402

INSTALLER = ROOT / "overlay" / "patch_cache_tail_evict.py"

POOL = '''\
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool."""
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
'''


class Block:
    def __init__(self, block_id: int, num_tokens: int | None, hashed: bool = True):
        self.block_id = block_id
        self.block_hash = ("h", block_id) if hashed else None
        self.block_hash_num_tokens = num_tokens
        self.prev_free_block = None
        self.next_free_block = None


class Queue:
    def __init__(self, blocks: list[Block]):
        self.fake_free_list_head = Block(-1, None, hashed=False)
        self.fake_free_list_tail = Block(-2, None, hashed=False)
        prev = self.fake_free_list_head
        for block in blocks:
            prev.next_free_block = block
            block.prev_free_block = prev
            prev = block
        prev.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = prev
        self.num_free_blocks = len(blocks)

    def remove(self, block: Block) -> None:
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def ids(self) -> list[int]:
        out = []
        block = self.fake_free_list_head.next_free_block
        while block is not self.fake_free_list_tail:
            out.append(block.block_id)
            block = block.next_free_block
        return out


def test_rank_prefers_unhashed_then_deeper_then_older():
    assert rank_key(False, None, 5) < rank_key(True, 100_000, 0)
    assert rank_key(True, 100_000, 9) < rank_key(True, 3584, 0)
    assert rank_key(True, 8192, 1) < rank_key(True, 8192, 4)
    assert rank_key(True, None, 0) == rank_key(True, 0, 0)


def test_paused_head_survives_a_newer_medium_page():
    # Queue order is what LRU would evict: the paused agent's tail, then its
    # first page, then a newer short request. Depth order takes the tail and
    # the short request, and leaves the first page.
    tail = Block(1, 100_352)
    head = Block(2, 3584)
    chaff = Block(3, 8192)
    queue = Queue([tail, head, chaff])
    taken = [block.block_id for block in select_blocks(queue, 2)]
    assert taken == [1, 3]
    assert queue.ids() == [2]
    assert queue.num_free_blocks == 1


def test_unhashed_goes_before_any_cached_page():
    cached = Block(7, 50_000)
    fresh = Block(8, None, hashed=False)
    queue = Queue([cached, fresh])
    taken = [block.block_id for block in select_blocks(queue, 1)]
    assert taken == [8]
    assert queue.ids() == [7]


def test_zero_blocks_does_not_touch_the_queue():
    queue = Queue([Block(1, 3584)])
    assert select_blocks(queue, 0) == []
    assert queue.ids() == [1]
    assert queue.num_free_blocks == 1


def _run(pool: Path, flag: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_CACHE_TAIL_EVICT"] = flag
    env["GLM53_BLOCK_POOL_PY"] = str(pool)
    return subprocess.run(
        [sys.executable, str(INSTALLER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_flag_off_does_not_read_a_missing_pool(tmp_path: Path):
    missing = tmp_path / "nope.py"
    result = _run(missing, "0")
    assert result.returncode == 0
    assert "unchanged" in result.stdout
    assert not missing.exists()


def test_flag_on_patches_once(tmp_path: Path):
    pool = tmp_path / "block_pool.py"
    pool.write_text(POOL)
    result = _run(pool, "1")
    assert result.returncode == 0, result.stderr
    text = pool.read_text()
    assert text.count("[glm53-cache-tail-evict]") == 2
    assert "popleft_n(num_blocks)" not in text
    assert "_glm53_tail_first_blocks(num_blocks)" in text
    again = _run(pool, "1")
    assert again.returncode == 0
    assert pool.read_text() == text


def test_drift_writes_nothing(tmp_path: Path):
    pool = tmp_path / "block_pool.py"
    pool.write_text(POOL.replace("popleft_n(num_blocks)", "popleft_n(num_blocks)  # moved"))
    before = pool.read_text()
    result = _run(pool, "1")
    assert result.returncode == 1
    assert pool.read_text() == before
