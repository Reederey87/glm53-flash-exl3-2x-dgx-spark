#!/usr/bin/env python3
"""
patch_glm5_drafter_group.py -- teach the GLM-5-Next KV layout about the
DFlash2 drafter's SlidingWindowSpec layers.

Edits vllm/v1/core/kv_cache_utils.py IN PLACE (build-time, inside the
radixark/vllm-glm53-flash sm121 image).

PROBLEM
-------
`_get_kv_cache_groups_glm5_next` returns None the moment any non-mamba /
non-tail spec is not exactly MLAAttentionSpec. The DFlash2 drafter registers 5
plain SlidingWindowSpec layers, so the whole model drops to the generic
uniform-page path -- which provably cannot serve GLM-5.3-Flash: page
unification rescales the kpool tail's block away from its pool size and boot
dies at warmup's `assert tail_kv_cache.shape[2] == pool_size` (see
~/lane1_fail6.log / ~/lane1_fail7.log on Reddie).

DESIGN
------
Keep the GLM-5-Next fast path bit-for-bit identical for the base model and
extend it with ONE extra group for the drafter, appended LAST (existing group
ids stay stable). Two modes, decided from the geometry:

  EXACT FIT (preferred; both deployed geometries land here): rescale the
  drafter's block size so its REAL page equals the MLA page exactly
  (block = mla_page // drafter_bytes_per_token), and let drafter layer i
  co-own MLA tensor i (`shared_by`) at disjoint block ids from the one shared
  BlockPool -- like mamba. The per-block byte cost of the pool is UNCHANGED,
  so KV capacity stays at the base model's; the sliding window bounds the
  drafter to a handful of block ids per request.

  CRITICAL, learned from boot 8 (~/lane1_fail8.log): `page_size_padded` is
  INVALID when the backend splits a large manager block into smaller kernel
  blocks (FlashInfer picked kernel 64 for a 2304-token manager; the strided
  path applied the full per-page stride to each KERNEL block -> OOB). Safe
  when manager block == kernel block (64): one kernel block per page, stride
  is the MLA page, view stays inside the page.

  Exact fit (no padding) is gated on:
    - mla_page divisible by the drafter's bytes/token;
    - fit block divisible by 64;
    - fit block and MLA block divide one another;
    - at most as many drafter layers as MLA tensors to ride in.
  656 B MLA vs 4096 B/token DFlash2 only exact-fits at MLA block 16384+.

  PADDED SLOT-SHARE (this geometry): compact manager block 64 +
  page_size_padded=mla_page. Drafter layer i co-owns MLA tensor i at
  disjoint window-bounded block ids (~49/req), like mamba. Per-block pool
  bytes unchanged. LCM(3584, 64)=3584.

  STANDALONE tensors (last resort): only if there are more drafter layers
  than MLA tensors to ride.

`_glm5_next_tensor_layout` detects the drafter group (uniform SWA) and
returns it as a 9th tuple element; the three consumers stay in lock-step:
  - `get_kv_cache_config_from_groups`: draft_page == mla_page (exact-fit or
    padded slot-share) -> drafter layer i joins MLA tensor i's shared_by;
    else standalone tensors + per-block cost;
  - `_pool_bytes_per_block`: standalone drafter pages only;
  - `_max_memory_usage_bytes_from_groups`: charges the drafter's window-
    bounded block-id demand at the per-block byte sum.

Runner-side audit (no edits needed there):
  - init_attn_backend builds per-group AttentionGroups generically; the
    drafter group's UniformTypeKVCacheSpecs unwraps to the per-layer SWA
    spec; prepare_kernel_block_sizes may pick a smaller kernel block --
    fine, both modes reshape through the contiguous path.
  - _reshape_kv_cache: num_blocks = raw.numel() // page_size_bytes is the
    pool's num_blocks in both modes (exact fit: the MLA tensor divided by
    mla_page; standalone: the compact tensor divided by draft_page).
  - _kv_first_layers_sharing_pool_with_mamba: blocks-first SWA backends
    report block_dim 0, so no page-aligned restride is triggered; the
    exact-fit contiguous view is already page-aligned per manager block.
  - Scheduler: generate_scheduler_kv_cache_config unwraps the group to a
    SlidingWindowSpec -> SlidingWindowManager; HybridKVCacheCoordinator's
    verify_and_split handles an extra participating spec group generically.
  - Speculator (dflash2): set_attn calls init_attn_backend with
    active_layer_names=draft layers; the drafter group id indexes
    BlockTables.input_block_tables generically.

Usage:
    python3 patch_glm5_drafter_group.py [--kv-file PATH] [--dry-run]

Idempotent: re-running on an already-patched file is a no-op (exit 0).
Fails loudly (AssertionError, nonzero exit) if any anchor is missing.
"""

from __future__ import annotations

import argparse
import ast
import sys

DEFAULT_KV_FILE = (
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py"
)

MARKER = "DFLASH2-DRAFTER-GROUP"

# ---------------------------------------------------------------------------
# Anchored edits. Every anchor must appear EXACTLY ONCE in the target file.
# ---------------------------------------------------------------------------

# -- _get_kv_cache_groups_glm5_next: partition drafter layers out ------------

EDIT_PARTITION_ANCHOR = """\
    attn_specs = {
        k: v
        for k, v in kv_cache_spec.items()
        if not isinstance(v, (MambaSpec, KpoolTailSpec))
    }
    if not mamba_specs or not all(
        type(s) is MLAAttentionSpec for s in attn_specs.values()
    ):
        return None
"""

EDIT_PARTITION_NEW = """\
    # DFLASH2-DRAFTER-GROUP: a spec-decode drafter (DFlash2) adds plain
    # SlidingWindowSpec layers on top of the GLM-5-Next hybrid. Partition them
    # out (exact type: KpoolTailSpec subclasses SlidingWindowSpec) so they do
    # not disqualify the model from this fast path; they are appended as one
    # extra group below.
    draft_specs = {
        k: v for k, v in kv_cache_spec.items() if type(v) is SlidingWindowSpec
    }
    attn_specs = {
        k: v
        for k, v in kv_cache_spec.items()
        if not isinstance(v, (MambaSpec, KpoolTailSpec))
        and type(v) is not SlidingWindowSpec
    }
    if not mamba_specs or not all(
        type(s) is MLAAttentionSpec for s in attn_specs.values()
    ):
        return None
"""

# -- _get_kv_cache_groups_glm5_next: build + append the drafter group --------

EDIT_GROUPS_RETURN_ANCHOR = """\
    mamba_grouped_names: list[list[str]] = [[] for _ in range(num_groups)]
    for k, name in enumerate(mamba_specs):
        mamba_grouped_names[k % num_groups].append(name)
    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
    )
"""

EDIT_GROUPS_RETURN_NEW = """\
    mamba_grouped_names: list[list[str]] = [[] for _ in range(num_groups)]
    for k, name in enumerate(mamba_specs):
        mamba_grouped_names[k % num_groups].append(name)

    # Drafter group (DFLASH2-DRAFTER-GROUP): one extra group for the spec-
    # decode drafter's SlidingWindowSpec layers, appended LAST so existing
    # group ids stay stable. NEVER page_size_padded: a padded spec routes the
    # runner into the strided-view reshape, which is invalid when the backend
    # virtually splits the manager block into smaller kernel blocks (boot 8:
    # FlashInfer picked kernel block 64 for a 2304-token manager block and
    # the per-KERNEL-block page stride blew past the tensor). Both modes
    # below use the ordinary contiguous reshape, valid under any split.
    draft_group = None
    if draft_specs:
        any_draft = next(iter(draft_specs.values()))
        assert all(spec == any_draft for spec in draft_specs.values()), (
            "drafter SlidingWindowSpec layers must share one spec"
        )
        draft_bytes_per_token = any_draft.page_size_bytes // any_draft.block_size
        mla_block = mla_specs[mla_names[0]].block_size
        fit_block = (
            mla_page // draft_bytes_per_token
            if mla_page % draft_bytes_per_token == 0
            else 0
        )
        if (
            fit_block
            # A 64-divisible manager block is divisible by every int kernel
            # block size the SWA backends register (16/32/64), so
            # select_common_block_size always finds a clean split.
            and fit_block % 64 == 0
            # Keep resolve_kv_cache_block_sizes' scheduler LCM at
            # max(mla_block, fit_block) instead of exploding.
            and (fit_block % mla_block == 0 or mla_block % fit_block == 0)
            and len(draft_specs) <= len(mla_names)
        ):
            # EXACT FIT: the drafter's real page equals the MLA page, so
            # drafter layer i co-owns MLA tensor i at disjoint block ids
            # (like mamba) with a contiguous view: kernel block j of manager
            # block b lands at b * mla_page + j * kernel_page, inside block
            # b's own page. Per-block pool cost unchanged.
            logger.info(
                "DFlash2 drafter KV: exact-fit block=%d mla_page=%d",
                fit_block,
                mla_page,
            )
            new_draft_specs: dict[str, KVCacheSpec] = {
                name: replace(s, block_size=fit_block)
                for name, s in draft_specs.items()
            }
        else:
            # PADDED SLOT-SHARE: 656 vs 4096 cannot exact-fill on this MLA
            # block. Manager 64 matches the SWA kernel, so padding the page
            # to mla_page is a safe strided view (boot 8 OOB was kernel 64
            # inside a 2304-token manager). Layer i co-owns MLA tensor i.
            compact_block = 64
            logger.info(
                "DFlash2 drafter KV: padded slot-share block=%d "
                "mla_page=%d (was block=%d); exact-fit page mismatch "
                "draft_bytes/token=%d",
                compact_block,
                mla_page,
                any_draft.block_size,
                draft_bytes_per_token,
            )
            new_draft_specs = {
                name: replace(
                    s,
                    block_size=compact_block,
                    page_size_padded=mla_page,
                )
                for name, s in draft_specs.items()
            }
        draft_uniform = UniformTypeKVCacheSpecs.from_specs(new_draft_specs)
        assert draft_uniform is not None
        draft_group = KVCacheGroupSpec(list(new_draft_specs), draft_uniform)

    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
        + ([draft_group] if draft_group is not None else [])
    )
"""

# -- _glm5_next_tensor_layout: return-type annotation ------------------------

EDIT_LAYOUT_ANNOT_ANCHOR = """\
        list[str],
        int,
    ]
    | None
):
"""

EDIT_LAYOUT_ANNOT_NEW = """\
        list[str],
        int,
        KVCacheGroupSpec | None,
    ]
    | None
):
"""

# -- _glm5_next_tensor_layout: docstring Returns -----------------------------

EDIT_LAYOUT_DOC_ANCHOR = """\
      - (attn_group, mamba_groups, mla_names, idx_names, mla_page, idx_page,
         tail_names, tail_page)
"""

EDIT_LAYOUT_DOC_NEW = """\
      - (attn_group, mamba_groups, mla_names, idx_names, mla_page, idx_page,
         tail_names, tail_page, draft_group)
"""

# -- _glm5_next_tensor_layout: detect the drafter group ----------------------

EDIT_LAYOUT_DETECT_ANCHOR = """\
    attn_group: KVCacheGroupSpec | None = None
    tail_group: KVCacheGroupSpec | None = None
    for g in uniform_groups:
        group_inner = cast(UniformTypeKVCacheSpecs, g.kv_cache_spec).kv_cache_specs
        if all(type(s) is MLAAttentionSpec for s in group_inner.values()):
            attn_group = g
        elif all(isinstance(s, KpoolTailSpec) for s in group_inner.values()):
            tail_group = g
"""

EDIT_LAYOUT_DETECT_NEW = """\
    attn_group: KVCacheGroupSpec | None = None
    tail_group: KVCacheGroupSpec | None = None
    draft_group: KVCacheGroupSpec | None = None
    for g in uniform_groups:
        group_inner = cast(UniformTypeKVCacheSpecs, g.kv_cache_spec).kv_cache_specs
        if all(type(s) is MLAAttentionSpec for s in group_inner.values()):
            attn_group = g
        elif all(isinstance(s, KpoolTailSpec) for s in group_inner.values()):
            tail_group = g
        elif group_inner and all(
            type(s) is SlidingWindowSpec for s in group_inner.values()
        ):
            # DFLASH2-DRAFTER-GROUP: the spec-decode drafter's SWA group
            # (validated below once mla_page is known).
            draft_group = g
"""

# -- _glm5_next_tensor_layout: validate the drafter group --------------------

EDIT_LAYOUT_VALIDATE_ANCHOR = """\
    if any(g.kv_cache_spec.page_size_bytes != mla_page for g in mamba_groups):
        return None
    tail_names: list[str] = []
"""

EDIT_LAYOUT_VALIDATE_NEW = """\
    if any(g.kv_cache_spec.page_size_bytes != mla_page for g in mamba_groups):
        return None
    if draft_group is not None:
        # DFLASH2-DRAFTER-GROUP: one uniform page across drafter layers.
        # Padded slot-share (page_size_padded=mla_page, block=64) is valid
        # because manager==kernel so the strided view does not split a page.
        # page == mla_page means slot-sharing of the MLA tensors
        # (needs one tensor per drafter layer); any other page means
        # standalone drafter tensors.
        draft_inner = cast(
            UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
        ).kv_cache_specs
        draft_pages = {s.page_size_bytes for s in draft_inner.values()}
        if len(draft_pages) != 1:
            return None
        if any(s.page_size_padded is not None for s in draft_inner.values()):
            if any(
                s.block_size != 64 or s.page_size_padded != mla_page
                for s in draft_inner.values()
            ):
                return None
        if (
            draft_pages.pop() == mla_page
            and len(draft_group.layer_names) > len(mla_names)
        ):
            return None
    tail_names: list[str] = []
"""

# -- _glm5_next_tensor_layout: return the drafter group ----------------------

EDIT_LAYOUT_RETURN_ANCHOR = """\
    return (
        attn_group,
        mamba_groups,
        mla_names,
        idx_names,
        mla_page,
        idx_pages.pop(),
        tail_names,
        tail_page,
    )
"""

EDIT_LAYOUT_RETURN_NEW = """\
    return (
        attn_group,
        mamba_groups,
        mla_names,
        idx_names,
        mla_page,
        idx_pages.pop(),
        tail_names,
        tail_page,
        draft_group,
    )
"""

# -- _pool_bytes_per_block: 9-tuple + standalone drafter bytes ---------------

EDIT_POOL_BYTES_ANCHOR = """\
        _, _, mla_names, idx_names, mla_page, idx_page, _, _ = glm5
        return len(mla_names) * mla_page + len(idx_names) * idx_page
"""

EDIT_POOL_BYTES_NEW = """\
        # DFLASH2-DRAFTER-GROUP: an exact-fit drafter (page == mla_page)
        # slot-shares the MLA tensors and adds no bytes; a standalone drafter
        # adds one page per drafter layer.
        _, _, mla_names, idx_names, mla_page, idx_page, _, _, draft_group = glm5
        per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        if draft_group is not None:
            draft_page = next(
                iter(
                    cast(
                        UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
                    ).kv_cache_specs.values()
                )
            ).page_size_bytes
            if draft_page != mla_page:
                per_block += len(draft_group.layer_names) * draft_page
        return per_block
"""

# -- get_kv_cache_config_from_groups: destructure + drafter mode -------------

EDIT_CONFIG_DESTRUCTURE_ANCHOR = """\
        (
            _,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _tail_page,
        ) = glm5n
"""

EDIT_CONFIG_DESTRUCTURE_NEW = """\
        (
            _,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _tail_page,
            draft_group,
        ) = glm5n
        draft_names: list[str] = []
        draft_page = 0
        draft_shared = False
        if draft_group is not None:
            draft_names = list(draft_group.layer_names)
            draft_page = next(
                iter(
                    cast(
                        UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
                    ).kv_cache_specs.values()
                )
            ).page_size_bytes
            # Exact fit: the drafter's real page equals the MLA page, so it
            # rides the MLA tensors; otherwise it gets standalone tensors.
            draft_shared = draft_page == mla_page
"""

# -- get_kv_cache_config_from_groups: per-block cost (standalone mode) -------

EDIT_CONFIG_PER_BLOCK_ANCHOR = """\
        per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        num_blocks = available_memory // per_block
"""

EDIT_CONFIG_PER_BLOCK_NEW = """\
        per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        if draft_names and not draft_shared:
            # DFLASH2-DRAFTER-GROUP (standalone): drafter tensors are part of
            # every block's byte cost.
            per_block += len(draft_names) * draft_page
        num_blocks = available_memory // per_block
"""

# -- get_kv_cache_config_from_groups: drafter co-owns MLA tensor i -----------

EDIT_CONFIG_SHARED_BY_ANCHOR = """\
                shared_by=[mla_name]
                + [g.layer_names[i] for g in mamba_groups if i < len(g.layer_names)],
            )
            for i, mla_name in enumerate(mla_names)
"""

EDIT_CONFIG_SHARED_BY_NEW = """\
                shared_by=[mla_name]
                + [g.layer_names[i] for g in mamba_groups if i < len(g.layer_names)]
                # DFLASH2-DRAFTER-GROUP (exact fit): drafter layer i rides MLA
                # tensor i (contiguous view, disjoint block ids), like mamba.
                + ([draft_names[i]] if draft_shared and i < len(draft_names) else []),
            )
            for i, mla_name in enumerate(mla_names)
"""

# -- get_kv_cache_config_from_groups: standalone drafter tensors -------------

EDIT_CONFIG_DRAFT_TENSORS_ANCHOR = """\
            KVCacheTensor(
                size=idx_page * num_blocks,
                shared_by=(
                    [idx_names[i], tail_names[i]] if tail_names else [idx_names[i]]
                ),
            )
            for i in range(len(idx_names))
        ]
"""

EDIT_CONFIG_DRAFT_TENSORS_NEW = """\
            KVCacheTensor(
                size=idx_page * num_blocks,
                shared_by=(
                    [idx_names[i], tail_names[i]] if tail_names else [idx_names[i]]
                ),
            )
            for i in range(len(idx_names))
        ] + [
            # DFLASH2-DRAFTER-GROUP (standalone): compact per-layer drafter
            # tensors; contiguous reshape, safe under kernel block splitting.
            KVCacheTensor(size=draft_page * num_blocks, shared_by=[name])
            for name in ([] if draft_shared else draft_names)
        ]
"""

# -- _max_memory_usage_bytes_from_groups: destructure ------------------------

EDIT_MAXMEM_DESTRUCTURE_ANCHOR = """\
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _tail_page,
        ) = glm5n
"""

EDIT_MAXMEM_DESTRUCTURE_NEW = """\
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _tail_page,
            draft_group,
        ) = glm5n
"""

# -- _max_memory_usage_bytes_from_groups: drafter demand + per-block ---------

EDIT_MAXMEM_BLOCKS_ANCHOR = """\
        if tail_names:
            # Tail: 1 block/req (KpoolTailSpec.max_admission_blocks_per_request
            # == 1), drawn from the shared pool.
            blocks_needed += 1
        return blocks_needed * (len(mla_names) * mla_page + len(idx_names) * idx_page)
"""

EDIT_MAXMEM_BLOCKS_NEW = """\
        if tail_names:
            # Tail: 1 block/req (KpoolTailSpec.max_admission_blocks_per_request
            # == 1), drawn from the shared pool.
            blocks_needed += 1
        per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        if draft_group is not None:
            # DFLASH2-DRAFTER-GROUP: charge the drafter's window-bounded
            # block-id demand; a standalone drafter also adds its pages to
            # every block's byte cost (an exact-fit one rides the MLA
            # tensors and adds none).
            draft_uniform = draft_group.kv_cache_spec
            assert isinstance(draft_uniform, UniformTypeKVCacheSpecs)
            blocks_needed += draft_uniform.max_memory_usage_pages(vllm_config)
            draft_page = next(
                iter(draft_uniform.kv_cache_specs.values())
            ).page_size_bytes
            if draft_page != mla_page:
                per_block += len(draft_group.layer_names) * draft_page
        return blocks_needed * per_block
"""

EDITS: list[tuple[str, str, str]] = [
    (
        "groups: partition drafter SlidingWindowSpec layers out",
        EDIT_PARTITION_ANCHOR,
        EDIT_PARTITION_NEW,
    ),
    (
        "groups: build + append drafter group (exact-fit / standalone)",
        EDIT_GROUPS_RETURN_ANCHOR,
        EDIT_GROUPS_RETURN_NEW,
    ),
    (
        "layout: return-type annotation gains draft_group",
        EDIT_LAYOUT_ANNOT_ANCHOR,
        EDIT_LAYOUT_ANNOT_NEW,
    ),
    (
        "layout: docstring Returns gains draft_group",
        EDIT_LAYOUT_DOC_ANCHOR,
        EDIT_LAYOUT_DOC_NEW,
    ),
    (
        "layout: detect drafter SWA uniform group",
        EDIT_LAYOUT_DETECT_ANCHOR,
        EDIT_LAYOUT_DETECT_NEW,
    ),
    (
        "layout: validate drafter (uniform page, never padded)",
        EDIT_LAYOUT_VALIDATE_ANCHOR,
        EDIT_LAYOUT_VALIDATE_NEW,
    ),
    (
        "layout: return draft_group (9th element)",
        EDIT_LAYOUT_RETURN_ANCHOR,
        EDIT_LAYOUT_RETURN_NEW,
    ),
    (
        "_pool_bytes_per_block: standalone drafter bytes",
        EDIT_POOL_BYTES_ANCHOR,
        EDIT_POOL_BYTES_NEW,
    ),
    (
        "config: destructure + drafter mode",
        EDIT_CONFIG_DESTRUCTURE_ANCHOR,
        EDIT_CONFIG_DESTRUCTURE_NEW,
    ),
    (
        "config: per-block cost includes standalone drafter",
        EDIT_CONFIG_PER_BLOCK_ANCHOR,
        EDIT_CONFIG_PER_BLOCK_NEW,
    ),
    (
        "config: exact-fit drafter layer i co-owns MLA tensor i",
        EDIT_CONFIG_SHARED_BY_ANCHOR,
        EDIT_CONFIG_SHARED_BY_NEW,
    ),
    (
        "config: standalone drafter tensors",
        EDIT_CONFIG_DRAFT_TENSORS_ANCHOR,
        EDIT_CONFIG_DRAFT_TENSORS_NEW,
    ),
    (
        "max-mem: destructure gains draft_group",
        EDIT_MAXMEM_DESTRUCTURE_ANCHOR,
        EDIT_MAXMEM_DESTRUCTURE_NEW,
    ),
    (
        "max-mem: charge drafter block-id demand + standalone bytes",
        EDIT_MAXMEM_BLOCKS_ANCHOR,
        EDIT_MAXMEM_BLOCKS_NEW,
    ),
]


def patch_file(path: str, dry_run: bool = False) -> int:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if MARKER in text:
        v4_old = (
            "        if any(s.page_size_padded is not None for s in draft_inner.values()):\n"
            "            return None\n"
        )
        v4_new = (
            "        if any(s.page_size_padded is not None for s in draft_inner.values()):\n"
            "            # Padded slot-share: manager block 64 so the kernel does not\n"
            "            # split the page (boot 8 OOB was kernel 64 in a 2304 manager).\n"
            "            if any(\n"
            "                s.block_size != 64 or s.page_size_padded != mla_page\n"
            "                for s in draft_inner.values()\n"
            "            ):\n"
            "                return None\n"
        )
        v3_marker = "padded slot-share block=%d"
        if v4_old in text:
            text = text.replace(v4_old, v4_new, 1)
            try:
                ast.parse(text, filename=path)
            except SyntaxError as e:
                raise AssertionError(
                    f"POST-EDIT ast.parse FAILED for {path}: {e}"
                ) from e
            if v3_marker in text:
                if dry_run:
                    print(f"[patch_glm5_drafter_group] DRY RUN -- {path} not written.")
                else:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(text)
                print(
                    f"[patch_glm5_drafter_group] {path}: padded slot-share v4 "
                    "(allow padded draft in glm5 layout) applied."
                )
                return 0
            # v3 grouping not yet present; keep going with mutated text.
        elif v3_marker in text:
            print(
                f"[patch_glm5_drafter_group] {path}: already patched "
                f"({MARKER} + padded slot-share); no-op."
            )
            return 0

        new_padded = (
            "            # PADDED SLOT-SHARE: 656 vs 4096 cannot exact-fill on this MLA\n"
            "            # block. Manager 64 matches the SWA kernel, so padding the page\n"
            "            # to mla_page is a safe strided view (boot 8 OOB was kernel 64\n"
            "            # inside a 2304-token manager). Layer i co-owns MLA tensor i.\n"
            "            compact_block = 64\n"
            "            logger.info(\n"
            "                \"DFlash2 drafter KV: padded slot-share block=%d \"\n"
            "                \"mla_page=%d (was block=%d); exact-fit page mismatch \"\n"
            "                \"draft_bytes/token=%d\",\n"
            "                compact_block,\n"
            "                mla_page,\n"
            "                any_draft.block_size,\n"
            "                draft_bytes_per_token,\n"
            "            )\n"
            "            new_draft_specs = {\n"
            "                name: replace(\n"
            "                    s,\n"
            "                    block_size=compact_block,\n"
            "                    page_size_padded=mla_page,\n"
            "                )\n"
            "                for name, s in draft_specs.items()\n"
            "            }\n"
        )

        # v3: compact-64 standalone already present (GHCR image + v2).
        v2_compact = (
            "            compact_block = 64\n"
            "            if any_draft.block_size > compact_block:\n"
        )
        if v2_compact in text:
            start = text.find("            # STANDALONE: compact per-layer tensors.")
            end = text.find("        draft_uniform = UniformTypeKVCacheSpecs.from_specs(new_draft_specs)")
            if start < 0 or end < 0 or end <= start:
                raise AssertionError(
                    f"{path}: {MARKER} + compact_block present but cannot "
                    "locate standalone block for padded slot-share v3"
                )
            text = text[:start] + new_padded + text[end:]
            try:
                ast.parse(text, filename=path)
            except SyntaxError as e:
                raise AssertionError(
                    f"POST-EDIT ast.parse FAILED for {path}: {e}"
                ) from e
            if dry_run:
                print(f"[patch_glm5_drafter_group] DRY RUN -- {path} not written.")
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
            print(
                f"[patch_glm5_drafter_group] {path}: padded slot-share v3 applied."
            )
            return 0

        # v2: shrink standalone DFlash pages off the 1152 MLA manager block.
        old_standalone = (
            "            # STANDALONE: the drafter's geometry cannot exactly fill the MLA\n"
            "            # page; keep its spec as-is and give its layers compact tensors\n"
            "            # of their own (emitted in get_kv_cache_config_from_groups and\n"
            "            # charged in the per-block cost).\n"
            "            new_draft_specs = dict(draft_specs)\n"
        )
        if old_standalone not in text:
            raise AssertionError(
                f"{path}: {MARKER} present but neither padded slot-share, "
                "compact-64, nor keep-as-is standalone block found"
            )
        text = text.replace(old_standalone, new_padded, 1)
        try:
            ast.parse(text, filename=path)
        except SyntaxError as e:
            raise AssertionError(
                f"POST-EDIT ast.parse FAILED for {path}: {e}"
            ) from e
        if dry_run:
            print(f"[patch_glm5_drafter_group] DRY RUN -- {path} not written.")
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        print(
            f"[patch_glm5_drafter_group] {path}: padded slot-share v3 applied "
            "(from keep-as-is)."
        )
        return 0

    # Sanity: the file we expect (guards against pointing at the wrong tree).
    for required in (
        "def _get_kv_cache_groups_glm5_next",
        "def _glm5_next_tensor_layout",
        "def _pool_bytes_per_block",
        "SlidingWindowSpec",
        "UniformTypeKVCacheSpecs",
    ):
        assert required in text, (
            f"ANCHOR PRECHECK FAILED: {required!r} not found in {path} -- "
            "is this really vllm/v1/core/kv_cache_utils.py?"
        )

    applied = []
    for name, anchor, replacement in EDITS:
        n = text.count(anchor)
        assert n == 1, (
            f"ANCHOR FAILED for edit [{name}]: expected exactly 1 occurrence, "
            f"found {n}. The upstream file has drifted -- re-derive the anchor "
            f"before building.\n--- anchor ---\n{anchor}\n--------------"
        )
        text = text.replace(anchor, replacement, 1)
        applied.append(name)

    # The patched source must still be valid Python.
    try:
        ast.parse(text, filename=path)
    except SyntaxError as e:
        raise AssertionError(f"POST-EDIT ast.parse FAILED for {path}: {e}") from e

    if dry_run:
        print(f"[patch_glm5_drafter_group] DRY RUN -- {path} not written.")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"[patch_glm5_drafter_group] {path}: {len(applied)} edits applied:")
    for name in applied:
        print(f"  - {name}")
    print("[patch_glm5_drafter_group] ast.parse OK.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--kv-file", default=DEFAULT_KV_FILE)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate anchors + parse, write nothing",
    )
    args = ap.parse_args()
    return patch_file(args.kv_file, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
