#!/usr/bin/env python3
"""Port the GLM-5.3 kpool tail correctness fixes.

Four independently verified defects make the one-block circular kpool tail
address the wrong memory. They ship as one unit because each one masks the
next: fixing only the stride leaves the wrong *logical* block, and forwarding
only positions turns a silent wrong-slot write into a CUDA-graph illegal
memory access.

1. ``kpool_compress.py`` -- the prefill seed kernel addressed tail blocks
   densely (``(blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM``) while the tail tensor
   aliases the indexer cache with the indexer's *padded* block stride
   (38016 B for GLM-5.3-Flash against a dense 2048 B tail block). Every
   prefill therefore wrote into an unrelated indexer block and left the
   request's own tail block untouched. The decode kernel in the same file
   already took ``stride(0)``/``stride(1)``; only the seed kernel did not.
   Port of vLLM #57477.

2. ``mamba_hybrid.py`` -- the hybrid model-state path called
   ``build_attn_metadata`` without ``positions=``, so
   ``KpoolTailMetadataBuilder`` saw ``positions is None`` and silently kept the
   *generic* paged mapping. That mapping indexes the tail group's one-block row
   by ``pos // 4`` against a 32-wide row, so every ``pos >= 128`` reads past the
   row. The correct mapping (``own_block * kpool + pos % kpool``) is already
   implemented and was simply never reached. ``default.py`` forwards positions;
   GLM-5.3 has KDA layers, so it always takes the hybrid path.

3. ``indexer.py`` -- once positions are forwarded,
   ``compute_kpool_tail_slot_mapping`` runs every step and returned
   ``slot_mapping.clone()``, a fresh allocation. CUDA graph capture records that
   transient address; on replay the tail kernels read a buffer that has since
   been freed or reused, which dies with ``Xid 13 Out Of Range Address`` about a
   second after graph capture. The function now writes into builder-owned
   persistent storage.

4. ``gpu/block_table.py`` -- the *live* V2 slot-mapping kernel loads the block
   table with no range term at all, so the tail group's ``pos // 4`` column
   index reads neighbouring requests' rows and, for the last request, past the
   whole allocation. Bound the column index; identity for every group whose row
   spans the sequence. The kit's ``patch_kpool_tail_slotmap.py`` clamps the
   *legacy* ``v1/worker/block_table.py`` kernel, which production does not
   execute -- the V2 runner imports only ``get_block_table_width`` from it.

Every target is preflighted and compiled before any write. Re-application is
idempotent, anchor drift fails closed, writes use atomic replacement, and stale
pyc files are removed.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


SITE = Path(
    os.environ.get(
        "GLM53_VLLM_SITE",
        "/usr/local/lib/python3.12/dist-packages/vllm",
    )
)

TARGETS = {
    "kpool": SITE / "models/glm5next/nvidia/ops/kpool_compress.py",
    "mamba": SITE / "v1/worker/gpu/model_states/mamba_hybrid.py",
    "indexer": SITE / "v1/attention/backends/mla/indexer.py",
    "blocktable": SITE / "v1/worker/gpu/block_table.py",
}


# ---------------------------------------------------------------------------
# 1. kpool_compress.py : address tail blocks by the padded indexer stride.
# ---------------------------------------------------------------------------

KPOOL_DOC_MARK = "    stride, so blocks are addressed through ``TAIL_BLOCK_ELEMS`` /"

KPOOL_DOC_ANCHOR = """    ``tail[block, {0:K, 1:score}, pos % KPOOL, :]``.
    \"\"\"
"""

KPOOL_DOC_PATCHED = """    ``tail[block, {0:K, 1:score}, pos % KPOOL, :]``.

    The tail cache aliases the indexer cache with the indexer's (padded) block
    stride, so blocks are addressed through ``TAIL_BLOCK_ELEMS`` /
    ``KPOOL_HEAD`` (``tail.stride(0)`` / ``tail.stride(1)``), never as a dense
    ``[num_blocks, 2, KPOOL, HEAD_DIM]`` array.
    \"\"\"
"""

KPOOL_SIG_ANCHOR = """    tslot_ptr,
    tail_ptr,
    n_tokens,
    HEAD_DIM: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
"""

KPOOL_SIG_PATCHED = """    tslot_ptr,
    tail_ptr,
    n_tokens,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
"""

KPOOL_BODY_ANCHOR = """    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM
    k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)
    s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)
    tl.store(tail_ptr + base + offs, k, mask=m)
    tl.store(tail_ptr + base + KPOOL * HEAD_DIM + offs, s, mask=m)
"""

KPOOL_BODY_PATCHED = """    base = blk * TAIL_BLOCK_ELEMS + (t % KPOOL) * HEAD_DIM
    k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)
    s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)
    tl.store(tail_ptr + base + offs, k, mask=m)
    tl.store(tail_ptr + base + KPOOL_HEAD + offs, s, mask=m)
"""

KPOOL_HOST_MARK = "    assert tail_kv_cache.ndim == 4 and tail_kv_cache.shape[1] == 2\n"

KPOOL_HOST_ANCHOR = """    \"\"\"Seed the paged tail cache from a prefill batch (see the kernel).\"\"\"
    assert tail_kv_cache.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
"""

KPOOL_HOST_PATCHED = """    \"\"\"Seed the paged tail cache from a prefill batch (see the kernel).\"\"\"
    assert tail_kv_cache.dtype == torch.bfloat16
    assert tail_kv_cache.ndim == 4 and tail_kv_cache.shape[1] == 2
    assert tail_kv_cache.stride(3) == 1 and tail_kv_cache.stride(2) == head_dim
    assert key.dtype == torch.bfloat16
"""

# No marker: the decode kernel's launch in this same file already passes
# ``TAIL_BLOCK_ELEMS=``/``KPOOL_HEAD=``, so that string is not unique. The
# seed launch is identified by the callee name instead.
KPOOL_CALL_ANCHOR = """    _kpool_tail_seed_kernel[(n,)](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        n,
        HEAD_DIM=head_dim,
"""

KPOOL_CALL_PATCHED = """    _kpool_tail_seed_kernel[(n,)](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        n,
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
        KPOOL_HEAD=tail_kv_cache.stride(1),
        HEAD_DIM=head_dim,
"""


# ---------------------------------------------------------------------------
# 2. mamba_hybrid.py : forward input positions to the metadata builder.
# ---------------------------------------------------------------------------

MAMBA_MARK = "            positions=input_batch.positions,\n"

MAMBA_ANCHOR = """            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=mamba_attn_metadata,
"""

MAMBA_PATCHED = """            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            model_specific_attn_metadata=mamba_attn_metadata,
"""


# ---------------------------------------------------------------------------
# 3. indexer.py : persistent, builder-owned tail slot-mapping storage.
# ---------------------------------------------------------------------------

INDEXER_SIG_MARK = "    out: torch.Tensor | None = None,\n"

INDEXER_SIG_ANCHOR = """    num_actual_tokens: int,
    num_reqs: int,
    kpool: int,
) -> torch.Tensor:
"""

INDEXER_SIG_PATCHED = """    num_actual_tokens: int,
    num_reqs: int,
    kpool: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
"""

INDEXER_DOC_ANCHOR = """    Pure torch (no Triton, no device sync): the indexer op consumes the tail
    slot mapping on its eager break, so the returned tensor need not be the
    persistent ``BlockTables`` buffer.
    \"\"\"
"""

INDEXER_DOC_PATCHED = """    Pure torch (no Triton, no device sync): the indexer op consumes the tail
    slot mapping on its eager break. When ``out`` is supplied the mapping is
    written into that caller-owned buffer, whose address must stay stable
    across CUDA graph replays; otherwise a fresh clone is returned.
    \"\"\"
"""

INDEXER_OUT_MARK = "    # [glm53-kpool-tail-correctness] Write into caller-owned persistent\n"

INDEXER_OUT_ANCHOR = """    out = slot_mapping.clone()
    if num_actual_tokens == 0:
        return out
"""

INDEXER_OUT_PATCHED = """    # [glm53-kpool-tail-correctness] Write into caller-owned persistent
    # storage. A fresh clone() per step is recorded by CUDA graph capture; on
    # replay the tail kernels read a buffer that has since been freed or
    # reused, which dies with Xid 13 Out Of Range Address right after capture.
    # The buffer is passed pre-viewed to this tensor's exact shape, so the
    # exported metadata is shape-identical to the clone() it replaces.
    if out is None:
        out = slot_mapping.clone()
    else:
        if out.shape != slot_mapping.shape:
            raise ValueError(
                "kpool tail slot-mapping buffer shape mismatch "
                f"({tuple(out.shape)} != {tuple(slot_mapping.shape)})"
            )
        out.copy_(slot_mapping)
    if num_actual_tokens == 0:
        return out
"""

INDEXER_INIT_MARK = "        self._tail_slot_mapping = torch.empty(\n"

INDEXER_INIT_ANCHOR = """        # No indexer-builder buffers (expanded_block_table / scheduler_metadata /
        # compressed_slot_mapping) -- the tail is storage-only and exports only
        # slot_mapping, which is rebuilt per step from the group's block table.
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
"""

INDEXER_INIT_PATCHED = """        # No indexer-builder buffers (expanded_block_table / scheduler_metadata /
        # compressed_slot_mapping) -- the tail is storage-only and exports only
        # slot_mapping, which is rebuilt per step from the group's block table.
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # [glm53-kpool-tail-correctness] Builder-owned persistent output. CUDA
        # graph replay retains the slot-mapping address, so returning a fresh
        # clone() every step leaves the captured kernels reading freed memory.
        self._tail_slot_mapping = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int64,
            device=device,
        )
"""

INDEXER_CALL_MARK = "            tail_slot_mapping = self._tail_slot_mapping[\n"

INDEXER_CALL_ANCHOR = """            slot_mapping = compute_kpool_tail_slot_mapping(
                slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
            )
"""

INDEXER_CALL_PATCHED = """            # [glm53-kpool-tail-correctness] Builder-owned persistent storage,
            # viewed to the batch shape so the exported metadata is unchanged.
            # ``view_as`` never copies, so the address captured by CUDA graphs
            # is the stable buffer base, not a per-step allocation.
            tail_slot_mapping = self._tail_slot_mapping[
                : slot_mapping.numel()
            ].view_as(slot_mapping)
            slot_mapping = compute_kpool_tail_slot_mapping(
                slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
                out=tail_slot_mapping,
            )
"""


# ---------------------------------------------------------------------------
# 4. gpu/block_table.py : bound the V2 slot-mapping column index.
# ---------------------------------------------------------------------------

BLOCKTABLE_MARK = "        in_range = block_indices < block_table_stride\n"

BLOCKTABLE_ANCHOR = """        block_indices = positions // (block_size * CP_SIZE)
        block_offsets = positions % (block_size * CP_SIZE)
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices
        )

        if CP_SIZE == 1:
            # Common case: Context parallelism is not used.
            slot_ids = block_numbers * block_size + block_offsets
        else:
            # Context parallelism is used.
            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            slot_ids = block_numbers * block_size + local_offsets
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)
"""

BLOCKTABLE_PATCHED = """        block_indices = positions // (block_size * CP_SIZE)
        block_offsets = positions % (block_size * CP_SIZE)
        # [glm53-kpool-tail-correctness] The kpool tail group's block-table row
        # is one block wide (padded to the block-table alignment), so its column
        # index runs off the end of the row -- and of the whole allocation for
        # the last request -- for every pos >= block_size * alignment. Bound the
        # column index; for any group whose row spans the sequence this is
        # identity, and the tail group's own builder supplies the real mapping.
        in_range = block_indices < block_table_stride
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices,
            mask=in_range,
            other=0,
        )

        if CP_SIZE == 1:
            # Common case: Context parallelism is not used.
            slot_ids = block_numbers * block_size + block_offsets
            slot_ids = tl.where(in_range, slot_ids, PAD_ID)
        else:
            # Context parallelism is used.
            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            slot_ids = block_numbers * block_size + local_offsets
            slot_ids = tl.where(is_local & in_range, slot_ids, PAD_ID)
"""


SITES = {
    "kpool": (
        ("seed docstring", KPOOL_DOC_MARK, KPOOL_DOC_ANCHOR, KPOOL_DOC_PATCHED),
        ("seed constexprs", None, KPOOL_SIG_ANCHOR, KPOOL_SIG_PATCHED),
        ("seed addressing", None, KPOOL_BODY_ANCHOR, KPOOL_BODY_PATCHED),
        ("seed layout asserts", KPOOL_HOST_MARK, KPOOL_HOST_ANCHOR, KPOOL_HOST_PATCHED),
        ("seed stride args", None, KPOOL_CALL_ANCHOR, KPOOL_CALL_PATCHED),
    ),
    "mamba": (
        ("forward positions", MAMBA_MARK, MAMBA_ANCHOR, MAMBA_PATCHED),
    ),
    "indexer": (
        ("mapping out param", INDEXER_SIG_MARK, INDEXER_SIG_ANCHOR, INDEXER_SIG_PATCHED),
        ("mapping docstring", None, INDEXER_DOC_ANCHOR, INDEXER_DOC_PATCHED),
        ("in-place write", INDEXER_OUT_MARK, INDEXER_OUT_ANCHOR, INDEXER_OUT_PATCHED),
        ("persistent buffer", INDEXER_INIT_MARK, INDEXER_INIT_ANCHOR, INDEXER_INIT_PATCHED),
        ("pass out buffer", INDEXER_CALL_MARK, INDEXER_CALL_ANCHOR, INDEXER_CALL_PATCHED),
    ),
    "blocktable": (
        ("bound column index", BLOCKTABLE_MARK, BLOCKTABLE_ANCHOR, BLOCKTABLE_PATCHED),
    ),
}


def verified_state(text: str, sites) -> bool:
    """True only when every site is present in its fully patched form.

    A site without a marker (a pure replacement) is verified by the absence of
    its anchor and the presence of exactly one patched block.
    """
    for _name, mark, anchor, patched in sites:
        if text.count(patched) != 1:
            return False
        if text.count(anchor) != patched.count(anchor):
            return False
        if mark is not None and text.count(mark) != 1:
            return False
    return True


def prepare(text: str, sites, label: str) -> tuple[str, str]:
    marks = sum(
        text.count(mark) for _name, mark, _anchor, _patched in sites if mark is not None
    )
    if marks:
        if not verified_state(text, sites):
            raise ValueError(
                f"partial/inconsistent {label} patch "
                f"(marks={marks}, sites={len(sites)})"
            )
        return text, "already present"

    if verified_state(text, sites):
        return text, "already patched"

    out = text
    for name, _mark, anchor, patched in sites:
        count = out.count(anchor)
        if count != 1:
            raise ValueError(
                f"pinned {label} anchor {name!r} drifted "
                f"(found {count}, expected 1)"
            )
        out = out.replace(anchor, patched, 1)

    if not verified_state(out, sites):
        raise ValueError(f"{label} post-patch verification failed")
    return out, "patched"


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
    for pyc in cache.glob(f"{target.stem}*.pyc"):
        pyc.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    preflight_only = "--preflight" in argv[1:]

    original: dict[str, str] = {}
    patched: dict[str, str] = {}
    actions: dict[str, str] = {}
    for label, target in TARGETS.items():
        if not target.is_file():
            raise SystemExit(f"missing {target}")
        original[label] = target.read_text()
        try:
            patched[label], actions[label] = prepare(
                original[label], SITES[label], label
            )
        except ValueError as exc:
            raise SystemExit(f"kpool tail correctness preflight failed: {exc}") from exc
        compile(patched[label], str(target), "exec")

    if preflight_only:
        print(
            "kpool tail correctness preflight OK "
            + " ".join(f"{name}={actions[name]}" for name in TARGETS)
        )
        return 0

    for label, target in TARGETS.items():
        if patched[label] != original[label]:
            replace_file(target, patched[label])
            clear_pyc(target)
    print(
        "kpool tail correctness "
        + " ".join(f"{name}={actions[name]}" for name in TARGETS)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
