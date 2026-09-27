#!/usr/bin/env python3
"""When GLM53_CACHE_TAIL_EVICT=1, evict the deepest cached block first.

Flag off (anything but the literal 1) prints a line and does not read or
write block_pool.py. Flag on replaces the one pop from the free list inside
get_new_blocks. Unhashed blocks still go first. Rollback is the flag at 0
and a new container: the installed file is the image copy.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MARK = "[glm53-cache-tail-evict]"
POOL = Path(
    os.environ.get(
        "GLM53_BLOCK_POOL_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py",
    )
)

POP_OLD = "        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)\n"
POP_NEW = (
    "        ret: list[KVCacheBlock] = "
    "self._glm53_tail_first_blocks(num_blocks)  # [glm53-cache-tail-evict]\n"
)
DEF_OLD = "    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:\n"
DEF_NEW = '''    def _glm53_tail_first_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """[glm53-cache-tail-evict] Deepest cached block first. See cache_tail_evict.py."""
        import sys as _glm53_sys

        if "/opt/glm53" not in _glm53_sys.path:
            _glm53_sys.path.insert(0, "/opt/glm53")
        from cache_tail_evict import select_blocks as _glm53_select_blocks

        return _glm53_select_blocks(self.free_block_queue, num_blocks)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
'''


def main() -> int:
    flag = os.environ.get("GLM53_CACHE_TAIL_EVICT", "0").strip()
    if flag != "1":
        print("GLM53_CACHE_TAIL_EVICT is off — installed sources unchanged", flush=True)
        return 0
    text = POOL.read_text()
    if MARK in text:
        print(f"{MARK} already present in {POOL}", flush=True)
        return 0
    if text.count(POP_OLD) != 1 or text.count(DEF_OLD) != 1:
        print(
            f"{MARK} anchor drift in {POOL}: "
            f"pop={text.count(POP_OLD)} def={text.count(DEF_OLD)}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    updated = text.replace(DEF_OLD, DEF_NEW, 1).replace(POP_OLD, POP_NEW, 1)
    if updated.count(MARK) < 2 or POP_OLD in updated:
        print(f"{MARK} replacement did not apply cleanly", file=sys.stderr, flush=True)
        return 1
    tmp = POOL.with_suffix(".py.glm53-tmp")
    tmp.write_text(updated)
    os.replace(tmp, POOL)
    print(f"{MARK} applied to {POOL}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
