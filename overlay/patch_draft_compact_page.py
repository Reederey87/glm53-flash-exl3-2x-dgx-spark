#!/usr/bin/env python3
"""Publish one drafter block id per compact page, instead of one per 64 tokens.

The DFlash2 group is a 5-layer sliding window (2,048) that slot-shares the
MLA page. The manager block is 64, so a cached 3,584-token segment holds 33
drafter ids and a running request holds ~89. Those ids do not extend the
MLA or KDA hit (the hybrid min already ignores this group). They occupy the
shared block-id pool.

This overlay, off unless GLM53_DRAFT_COMPACT_PAGE=1, raises that manager
block to the largest multiple of 64 that:

  * divides the MLA block, so the scheduler LCM stays the MLA page;
  * fits inside the MLA page at this boot's draft bytes/token;
  * the drafter backend accepts as a whole page.

A padded page must not be kernel-split. FlashInfer did that on an earlier
boot (kernel 64 inside a 2,304-token manager) and the strided view walked
off the tensor. The floor stays 64 when no larger page qualifies, and when a
sliding-window layer belongs to the target stack (its name contains
``language_model``). The drafter is loaded with an empty prefix, so its
layers are ``model.layers.*``, not an ``eagle_head`` tag.

DFlash context KV is a projection of the target hidden state, so the EAGLE
last-block drop is the wrong lookup once a page is hundreds of tokens
(vLLM #54092, #54163). The drop stays at the 64-token floor. Above that,
this group looks up to the reconciled boundary. It is not put back into
the hybrid min().

Runs after patch_glm5_drafter_group.py and patch_hybrid_prefix_hit.py.
Flag off leaves both files byte-identical.
"""
from __future__ import annotations

import os
import sys

KV_FILE = os.environ.get(
    "GLM53_KV_CACHE_UTILS_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py",
)
COORD_FILE = os.environ.get(
    "GLM53_KV_COORDINATOR_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py",
)
ENV_NAME = "GLM53_DRAFT_COMPACT_PAGE"
MARK = "[glm53-draft-page]"
FLOOR = 64

# The single `compact_block = 64` the drafter-group patch writes into the
# padded slot-share branch.
BLOCK_OLD = "            compact_block = 64\n"
BLOCK_NEW = """\
            compact_block = 64
            # [glm53-draft-page] One id per compact page. Floor 64 is today's
            # reservation. A larger page has to divide the MLA block, fit in
            # the MLA page, and be accepted whole by FlashAttention (the
            # drafter backend). A padded page that the backend would split
            # walks off the tensor, so those candidates stay at the floor.
            # Drafter layers are not tagged eagle_head: get_model() is called
            # with an empty prefix. Target layers live under language_model
            # and are MLA, so a sliding-window name that contains that path
            # is not this drafter and stays at the floor.
            _glm53_names = list(draft_specs)
            _glm53_reason = "floor"
            if any("language_model" in str(n) for n in _glm53_names):
                _glm53_reason = "non-drafter sliding-window layer"
            elif (
                _glm53_names
                and mla_block % 64 == 0
                and draft_bytes_per_token > 0
            ):
                _glm53_steps = mla_block // 64
                for _glm53_k in range(_glm53_steps, 1, -1):
                    if _glm53_steps % _glm53_k != 0:
                        continue
                    _glm53_cand = 64 * _glm53_k
                    if _glm53_cand * draft_bytes_per_token > mla_page:
                        continue
                    try:
                        from vllm.v1.attention.backends.flash_attn import (
                            FlashAttentionBackend as _GLM53FA,
                        )
                        from vllm.v1.worker.utils import (
                            select_common_block_size as _glm53_sel,
                        )
                        _glm53_kernel = _glm53_sel(_glm53_cand, [_GLM53FA])
                        _glm53_ok = _glm53_kernel == _glm53_cand
                    except Exception as _glm53_exc:
                        _glm53_kernel = None
                        _glm53_ok = False
                        _glm53_reason = f"backend check failed: {_glm53_exc}"
                    if not _glm53_ok:
                        if _glm53_reason == "floor":
                            _glm53_reason = (
                                f"backend would split {_glm53_cand} "
                                f"into {_glm53_kernel}"
                            )
                        continue
                    compact_block = _glm53_cand
                    _glm53_reason = "unsplit"
                    break
            logger.info(
                "DFlash2 drafter KV: compact page chose block=%d (%s) "
                "draft_bytes/token=%d mla_block=%d layers=%s",
                compact_block,
                _glm53_reason,
                draft_bytes_per_token,
                mla_block,
                _glm53_names,
            )
"""

VALID_OLD = """\
            if any(
                s.block_size != 64 or s.page_size_padded != mla_page
                for s in draft_inner.values()
            ):
                return None
"""
VALID_NEW = """\
            # [glm53-draft-page] Padded slot-share may use the compact page.
            # The page still has to be the MLA page, a multiple of the
            # 64-token floor, no bigger than that page, and a divisor of the
            # MLA block (scheduler LCM).
            if any(
                s.page_size_padded != mla_page
                or s.block_size % 64 != 0
                or s.block_size <= 0
                or s.unpadded_page_size_bytes > mla_page
                for s in draft_inner.values()
            ):
                return None
            _glm53_mla_block = 0
            if attn_group is not None:
                _glm53_mla_block = next(
                    iter(
                        cast(
                            UniformTypeKVCacheSpecs, attn_group.kv_cache_spec
                        ).kv_cache_specs.values()
                    )
                ).block_size
            if any(
                _glm53_mla_block % s.block_size != 0
                for s in draft_inner.values()
            ):
                return None
"""

DROP_OLD = "                drop_eagle_block = use_eagle and idx not in eagle_verified\n"
DROP_NEW = """\
                drop_eagle_block = use_eagle and idx not in eagle_verified
                if (
                    _glm53_is_draft_swa_spec(spec)
                    and self.single_type_managers[first_group_id].block_size > 64
                ):  # [glm53-draft-page]
                    # Context KV is a projection of the target hidden state.
                    # Dropping one compact page empties the drafter hit. The
                    # 64-token floor keeps the drop. This group stays out of
                    # the hybrid min().
                    drop_eagle_block = False
"""


def choose_draft_block(
    mla_block: int,
    mla_page: int,
    draft_bytes: int,
    accepts,
    *,
    names: tuple[str, ...] = ("model.layers.0.self_attn",),
) -> int:
    """Largest safe drafter manager block. ``accepts(block)`` is true when
    the backend will run that page unsplit. Anything that fails a check
    returns the 64-token floor.
    """
    if (
        not names
        or any("language_model" in str(name) for name in names)
        or mla_block < FLOOR
        or mla_block % FLOOR != 0
        or mla_page <= 0
        or draft_bytes <= 0
    ):
        return FLOOR
    steps = mla_block // FLOOR
    for k in range(steps, 1, -1):
        if steps % k != 0:
            continue
        block = FLOOR * k
        if block * draft_bytes > mla_page:
            continue
        if accepts(block):
            return block
    return FLOOR


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    found = text.count(old)
    if found != 1:
        raise SystemExit(f"{label}: expected 1 match, found {found}")
    return text.replace(old, new, 1)


def patch_kv(text: str) -> str:
    if MARK in text and "compact page chose block=%d" in text and "unpadded_page_size_bytes" in text:
        return text
    text = _replace_once(text, BLOCK_OLD, BLOCK_NEW, "compact-block")
    text = _replace_once(text, VALID_OLD, VALID_NEW, "padded-validator")
    compile(text, KV_FILE, "exec")
    return text


def patch_coord(text: str) -> str:
    if MARK in text and "block_size > 64" in text:
        return text
    if "def _glm53_is_draft_swa_spec(" not in text:
        raise SystemExit(f"{COORD_FILE}: hybrid drafter helper is missing")
    text = _replace_once(text, DROP_OLD, DROP_NEW, "eagle-drop")
    compile(text, COORD_FILE, "exec")
    return text


def _armed() -> bool:
    return os.environ.get(ENV_NAME, "0") == "1"


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def main() -> int:
    if not _armed():
        print(f"{MARK} {ENV_NAME} is off — files unchanged")
        return 0
    for path in (KV_FILE, COORD_FILE):
        if not os.path.isfile(path):
            raise SystemExit(f"missing {path}")
    with open(KV_FILE, encoding="utf-8") as fh:
        kv = fh.read()
    with open(COORD_FILE, encoding="utf-8") as fh:
        coord = fh.read()
    kv_out = patch_kv(kv)
    coord_out = patch_coord(coord)
    if kv_out != kv:
        _write(KV_FILE, kv_out)
        print(f"{MARK} patched {KV_FILE}")
    else:
        print(f"{MARK} {KV_FILE} already patched")
    if coord_out != coord:
        _write(COORD_FILE, coord_out)
        print(f"{MARK} patched {COORD_FILE}")
    else:
        print(f"{MARK} {COORD_FILE} already patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
