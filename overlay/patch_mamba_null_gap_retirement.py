#!/usr/bin/env python3
"""Retire obsolete align-mode Mamba states across null block gaps.

The pinned fork's generic ``_remove_blocks_in_range`` stops at the first null
block. Align-mode prefill can intentionally leave null gaps between recurrent
state blocks, so older states before a gap remain referenced indefinitely.

This installer adds a MambaManager override that skips null gaps, tracks the
already-retired prefix per request, and clears that tracking when the request is
freed. It is adapted from vLLM PR #55450 to the exact production fork, whose
MambaManager does not have mainline's ``_num_checkpoint_blocks`` member.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_SINGLE_TYPE_KV_MANAGER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/"
        "single_type_kv_cache_manager.py",
    )
)
MARK = "# [glm53-mamba-null-gap]"

INIT_OLD = """\
            self.last_state_block_idx: dict[str, int] = {}
            # The set of the requests that have been allocated blocks
"""
INIT_NEW = """\
            self.last_state_block_idx: dict[str, int] = {}
            # Highest align-mode block prefix already scanned for retirement.
            self._num_retired_blocks: dict[str, int] = {}  # [glm53-mamba-null-gap]
            # The set of the requests that have been allocated blocks
"""

REMOVE_ANCHOR = """\
    def remove_skipped_blocks(
        self,
        request_id: str,
"""
REMOVE_OVERRIDE = """\
    def _remove_blocks_in_range(
        self, request_id: str, first_block: int, last_block: int
    ) -> None:
        if self.mamba_cache_mode != "align":
            return super()._remove_blocks_in_range(
                request_id, first_block, last_block
            )
        blocks = self.req_to_blocks.get(request_id, [])
        first_block = max(
            first_block, self._num_retired_blocks.get(request_id, 0)
        )
        last_block = min(last_block, len(blocks))
        if first_block >= last_block:
            return
        freed: list[KVCacheBlock] = []
        # Align-mode prefill can leave null gaps between state checkpoints.
        for i in range(last_block - 1, first_block - 1, -1):
            if blocks[i].is_null:
                continue
            freed.append(blocks[i])
            blocks[i] = self._null_block
        if freed:
            self.block_pool.free_blocks(freed)
        self._num_retired_blocks[request_id] = last_block

"""

FREE_OLD = """\
            self.last_state_block_idx.pop(request_id, None)
            self._producer_partial_tail_reqs.pop(request_id, None)
"""
FREE_NEW = """\
            self.last_state_block_idx.pop(request_id, None)
            self._num_retired_blocks.pop(request_id, None)  # [glm53-mamba-null-gap]
            self._producer_partial_tail_reqs.pop(request_id, None)
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{TARGET}: expected one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def validate(text: str) -> None:
    ast.parse(text, filename=str(TARGET))
    expected = (
        ("Mamba retirement init", INIT_NEW),
        ("Mamba retirement override", REMOVE_OVERRIDE),
        ("Mamba retirement cleanup", FREE_NEW),
    )
    for label, snippet in expected:
        count = text.count(snippet)
        if count != 1:
            raise SystemExit(f"{TARGET}: expected exact {label}, found {count}")

    class_pos = text.find("class MambaManager(")
    next_class = text.find("\nclass ", class_pos + 1)
    class_text = text[class_pos : next_class if next_class >= 0 else len(text)]
    for label, snippet in expected:
        if snippet not in class_text:
            raise SystemExit(f"{TARGET}: {label} is outside MambaManager")
    if text.count(MARK) != 2:
        raise SystemExit(f"{TARGET}: null-gap markers missing or duplicated")


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"missing target: {TARGET}")
    text = TARGET.read_text()
    if MARK in text:
        validate(text)
        print("glm53: Mamba null-gap retirement already installed", file=sys.stderr)
        return 0

    text = replace_once(text, INIT_OLD, INIT_NEW, "MambaManager init")
    class_pos = text.find("class MambaManager(")
    remove_pos = text.find(REMOVE_ANCHOR, class_pos)
    if class_pos < 0 or remove_pos < 0:
        raise SystemExit(f"{TARGET}: MambaManager remove_skipped_blocks anchor missing")
    text = text[:remove_pos] + REMOVE_OVERRIDE + text[remove_pos:]
    text = replace_once(text, FREE_OLD, FREE_NEW, "MambaManager cleanup")
    validate(text)
    TARGET.write_text(text)
    print("glm53: installed Mamba null-gap retirement", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
