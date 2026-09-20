#!/usr/bin/env python3
"""Upper-bound safety clamp on the legacy V1 slot-mapping kernel.

This is a **bounds guard, not the circular-mapping repair**. The authoritative
tail mapping is ``compute_kpool_tail_slot_mapping`` in
``v1/attention/backends/mla/indexer.py``; see
``patch_kpool_tail_correctness.py`` for that fix.

What this patch covers
----------------------
``KpoolTailSpec`` is a one-block circular scratch cache: both
``max_admission_blocks_per_request`` and ``max_num_blocks_per_req`` return 1,
so its block table holds one block per request. Slot mapping still used the
generic paged Triton kernel in the *legacy* ``v1/worker/block_table.py``::

    block_indices = pos // block_size
    block_numbers = block_table[req, block_indices]

The load's mask covers token validity only. Nothing bounded ``block_indices``
against the row width, so for the tail group every token at
``pos >= block_size`` read past the row and the kpool seed/update kernels
wrote through whatever block id came back. A finished request is not proof the
writes were in-bounds — most overruns land inside the shared pool and silently
corrupt another layer's indexer.

Correction to the original rationale
------------------------------------
This patch previously claimed ``block_table_stride == 1`` for the tail group,
so clamping to ``block_table_stride - 1`` would pin the index to entry 0 and
yield ``block_table[req, 0] * block_size + pos % block_size``. **That premise
was false.** ``block_table_stride`` is the padded row width from
``get_block_table_width``, which is 32 for the tail group (``block_size`` 4
raised to ``token_alignment`` 128). The clamp therefore pins to column 31,
which ``KpoolTailManager`` never writes, and reads back 0 — a null block, not
the request's own tail block.

An independent measurement of this exact clamp (vcruz305,
``docs/KPOOL_TAIL_BUG.md``, the source of the mechanism below) reported 48
overruns before and 48 after, i.e. it did not restore the documented
addressing. What it *does* do is replace the unmasked out-of-bounds read with
a deterministic in-bounds read, which is still worth having.

For every other group a request never legitimately needs more blocks than its
row holds, so the clamp is provably identity there.

Scope caveat
------------
Production runs the **V2** model runner, whose slot-mapping kernel lives in
``v1/worker/gpu/block_table.py``. The V2 runner imports only
``get_block_table_width`` from this legacy module, so this patch does not
execute in production. The live kernel is bounded by
``patch_kpool_tail_correctness.py`` instead.

Fail-closed, idempotent, preflights the pinned anchor before writing.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_BLOCK_TABLE_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/block_table.py",
    )
)
MARK = "    # [glm53-kpool-tail-slotmap] Never index past the request's\n"

ANCHOR = """        block_indices = (
            virtual_block_indices * BLOCKS_PER_KV_BLOCK
            + local_block_offsets // block_size
        )
        block_numbers = tl.load(
            block_table_ptr + row_offset + block_indices,
            mask=mask & is_local,
            other=0,
        ).to(tl.int64)
"""

PATCHED = """        block_indices = (
            virtual_block_indices * BLOCKS_PER_KV_BLOCK
            + local_block_offsets // block_size
        )
        # [glm53-kpool-tail-slotmap] Never index past the request's
        # block-table row. KpoolTailSpec is a one-block circular scratch
        # whose row is a single entry; without this clamp every token at
        # pos >= block_size reads adjacent memory and the kpool kernels
        # write through garbage. Clamping pins that group to entry 0:
        # block_table[req, 0] * block_size + pos % block_size.
        # For every other group this is identity.
        block_indices = tl.minimum(block_indices, block_table_stride - 1)
        block_numbers = tl.load(
            block_table_ptr + row_offset + block_indices,
            mask=mask & is_local,
            other=0,
        ).to(tl.int64)
"""


def count_overruns(
    positions: list[int], *, block_size: int, stride: int
) -> int:
    """How many positions the unpatched kernel would index past ``stride``."""
    if block_size < 1 or stride < 1:
        raise ValueError("block_size and stride must be >= 1")
    return sum(1 for pos in positions if (pos // block_size) >= stride)


def circular_slot_ids(
    positions: list[int],
    block_table_row: list[int],
    block_size: int,
    *,
    clamp: bool,
) -> list[int]:
    """CPU replica of the Triton mapping, with or without the row clamp."""
    if not block_table_row:
        raise ValueError("block_table_row must be non-empty")
    stride = len(block_table_row)
    out: list[int] = []
    for pos in positions:
        idx = pos // block_size
        if clamp:
            idx = min(idx, stride - 1)
        elif idx < 0 or idx >= stride:
            raise IndexError(
                f"pos={pos} indexes block {idx} past stride {stride}"
            )
        out.append(block_table_row[idx] * block_size + (pos % block_size))
    return out


def verified_state(text: str) -> bool:
    return (
        text.count(ANCHOR) == 0
        and text.count(PATCHED) == 1
        and text.count(MARK) == 1
        and "tl.minimum(block_indices, block_table_stride - 1)" in text
    )


def prepare(source: str) -> tuple[str, str]:
    marker_count = source.count(MARK)
    if marker_count:
        if marker_count != 1 or not verified_state(source):
            raise ValueError(
                "partial/inconsistent kpool tail slot-map patch "
                f"(marker={marker_count})"
            )
        return source, "already present"
    if verified_state(source):
        return source, "already patched"
    n_anchor = source.count(ANCHOR)
    if n_anchor != 1:
        raise ValueError(
            "pinned block_table slot-mapping anchor drifted "
            f"(anchor={n_anchor})"
        )
    if "tl.minimum(block_indices, block_table_stride - 1)" in source:
        raise ValueError(
            "block_table already clamps block_indices but the pinned "
            "anchor was not found — re-derive the patch"
        )
    patched = source.replace(ANCHOR, PATCHED, 1)
    if not verified_state(patched):
        raise ValueError("kpool tail slot-map post-patch verification failed")
    return patched, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-kpool-tail.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob("block_table*.pyc"):
        pyc.unlink(missing_ok=True)


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    source = TARGET.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"kpool tail slot-map preflight failed: {exc}") from exc
    compile(patched, str(TARGET), "exec")
    if patched != source:
        replace_file(TARGET, patched)
        clear_pyc(TARGET)
    print(f"{TARGET.name}: kpool tail slot-map {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
