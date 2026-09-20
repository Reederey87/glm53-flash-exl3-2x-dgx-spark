#!/usr/bin/env python3
"""[glm53-apc-tail-floor] Floor the prefix-cache tail registration from (n - 1),
not n, so the deepest reusable state lands where a lookup can actually ask.

Task 45, site 1 of the hybrid + EAGLE prefix-cache defect cluster. Measured on
this cluster before the fix (`scripts/probe_apc_boundary_reachability.py`):
an exact replay of a 10,752-token prompt reused **7,168** tokens against a
reachable ceiling of **10,688** — one full 3,584-token hybrid page short, and
2.673 s instead of 0.251 s (11x). 7,168 and 14,336 behaved identically.

Why the off-by-one is structural, not a typo
--------------------------------------------
`KVCacheManager.get_computed_blocks` sets

    max_cache_hit_length = request.num_tokens - 1

because the last token must be recomputed to obtain logits. So for a request of
`n` prompt tokens the deepest *lookupable* hash unit is
`floor((n - 1) / hash_block_size) * hash_block_size`. Three sites that register
reusable state floored from `n` instead, and whenever `n` lands on the hash grid
they registered state one unit **above** any position a lookup can request:

    n = 10,752 (a multiple of 64 and of the 3,584-token page)
      registered at  10,752   <- unreachable
      lookupable to  10,688
      result:        the whole tail page is recomputed

    n = 6,500 (not a multiple of 64)
      registered at   6,464
      lookupable to    6,464   <- same value, which is why the defect hides
      result:        full reuse

The measured ladder shows exactly that split: every prompt length that is not a
multiple of the hash unit already reaches its ceiling, and every one that is
falls back a whole page. A prompt that is an exact multiple of the page size is
worse: `_cache_partial_tail_block` refuses a block-aligned tail outright
(`num_tokens % block_size == 0`), so no partial entry is registered at all.

Why this is safe, and why it is arithmetic rather than a heuristic
-----------------------------------------------------------------
The published entry is keyed by the token prefix `[0, q)` and holds the state
after exactly `q` tokens, for `q = floor((n - 1) / unit) * unit`. A consumer
that resumes at `q` is given a state that matches the position its own key
proves. The change therefore moves a registration to a *reachable* position
whose state is already materialized at that position; it does not invent reuse,
relax a key, or weaken the proof. It is the same value for every prompt length
that already worked (shown above), so the only lengths whose behaviour changes
are the ones that were broken.

The scheduler stop is **additional**, not a replacement: the existing
block-aligned stops and the shared-prefix junction stop are untouched, so no
intermediate chunk can end off-grid (which is the state-poisoning failure mode
that replacing those stops re-triggers).

Gated by `GLM53_APC_TAIL_FLOOR` (default 0 = byte-neutral). Idempotent via the
marker; every edit is exactly-once, ast-validated, and written atomically.
Fails closed on anchor drift rather than guessing.
"""

from __future__ import annotations

import ast
import os
import sys

MARKER = "[glm53-apc-tail-floor]"

VLLM = os.environ.get(
    "GLM53_VLLM_SITE",
    "/usr/local/lib/python3.12/dist-packages/vllm",
)

# --- site 1: the scheduler's mamba partial-tail stop ------------------------
SCHED_REL = "v1/core/sched/scheduler.py"
SCHED_OLD = """        tail_boundary = (
            request.num_prompt_tokens // self.hash_block_size * self.hash_block_size
            if self.mamba_partial_cache_hit
            else 0
        )
"""
SCHED_NEW = """        tail_boundary = (
            # [glm53-apc-tail-floor] Floor from (n - 1): the finder is capped at
            # `num_tokens - 1` (the last token is recomputed for logits), so a
            # boundary registered at `n` is one unit above anything a lookup can
            # request and never serves a hit.
            (request.num_prompt_tokens - 1) // self.hash_block_size
            * self.hash_block_size
            if self.mamba_partial_cache_hit
            else 0
        )
"""

# --- site 2: full attention's partial tail ----------------------------------
MGR_REL = "v1/core/single_type_kv_cache_manager.py"
FA_OLD = """        hash_block_size = self.block_pool.hash_block_size
        boundary_tokens = request.num_prompt_tokens // hash_block_size * hash_block_size
        if boundary_tokens == 0 or boundary_tokens > num_tokens:
"""
FA_NEW = """        hash_block_size = self.block_pool.hash_block_size
        # [glm53-apc-tail-floor] Same (n - 1) floor as the mamba tail: with
        # fine-grained hashing this is the deepest tail a replay can reach, and
        # it is the position the hybrid intersection needs to meet.
        boundary_tokens = (
            (request.num_prompt_tokens - 1) // hash_block_size * hash_block_size
        )
        if boundary_tokens == 0 or boundary_tokens > num_tokens:
"""

# --- site 3: the mamba partial tail ----------------------------------------
MAMBA_OLD = """        latest_prompt_hash_boundary = (
            request.num_prompt_tokens // hash_block_size
        ) * hash_block_size
"""
MAMBA_NEW = """        latest_prompt_hash_boundary = (  # [glm53-apc-tail-floor]
            (request.num_prompt_tokens - 1) // hash_block_size
        ) * hash_block_size
"""


def patch_text(text: str, rel: str, edits: list[tuple[str, str]]) -> str:
    """Apply every edit exactly once; refuse a partial or drifted file."""
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            raise SystemExit(
                f"{rel}: expected exactly one anchor, found {n}: "
                f"{old.strip().splitlines()[0][:70]!r}"
            )
        text = text.replace(old, new, 1)
    return text


def patch_file(path: str, rel: str, edits: list[tuple[str, str]]) -> int:
    with open(path, encoding="utf-8") as f:
        text = f.read()

    if MARKER in text:
        # Idempotency means "every edit fully present", not "marker seen".
        if all(new in text for _, new in edits):
            ast.parse(text, filename=path)
            print(f"[patch_apc_tail_boundary] {rel}: already patched; no-op.")
            return 0
        raise AssertionError(
            f"{rel}: marker present but the patched forms are incomplete "
            "(partial or interrupted earlier edit?); refusing to touch it."
        )

    text = patch_text(text, rel, edits)
    try:
        ast.parse(text, filename=path)
    except SyntaxError as e:
        raise AssertionError(f"POST-EDIT ast.parse FAILED for {rel}: {e}") from e

    tmp = path + ".glm53-apc-tail-floor.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)  # atomic: never leave a marker-bearing partial file
    print(f"[patch_apc_tail_boundary] {rel}: tail registration floored to (n - 1).")
    return 1


def main() -> int:
    if os.environ.get("GLM53_APC_TAIL_FLOOR", "0") != "1":
        print(
            "[patch_apc_tail_boundary] GLM53_APC_TAIL_FLOOR != 1 - skipping "
            "(stock tail registration)."
        )
        return 0

    targets = [
        (os.path.join(VLLM, SCHED_REL), SCHED_REL, [(SCHED_OLD, SCHED_NEW)]),
        (
            os.path.join(VLLM, MGR_REL),
            MGR_REL,
            [(FA_OLD, FA_NEW), (MAMBA_OLD, MAMBA_NEW)],
        ),
    ]
    for path, rel, edits in targets:
        if not os.path.isfile(path):
            # Enabled but nothing to patch: fail closed so `set -e` stops the
            # rank before `exec vllm serve` instead of booting silently stock.
            print(
                f"[patch_apc_tail_boundary] FATAL: target not found: {path}",
                file=sys.stderr,
            )
            return 1
        patch_file(path, rel, edits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
