#!/usr/bin/env python3
"""Stop the mamba block-alignment split from zeroing a sub-block prefill chunk.

Found by the codex pre-ship review of the adaptive-LPTT design (2026-09-01) and
verified against the running container.

In `Scheduler._mamba_block_aligned_split` the truncation reads:

    aligned_end = end // block_size * block_size
    if aligned_end > start or block_size <= max_prefill_tokens:
        end = aligned_end

`max_prefill_tokens = min(max_num_scheduled_tokens, long_prefill_token_threshold)`.
When LPTT >= block_size the SECOND disjunct is always true, so `end` is forced to
`aligned_end` even when `aligned_end <= start` -- i.e. when no block boundary lies
in (start, end]. The method then returns `max(end - start, 0)` == 0 and BOTH call
sites skip the request for that step (the running loop's reason #4, the waiting
loop's `if num_new_tokens == 0: break`).

That is not a crash; it is a total starvation stall, and it silently nullifies the
decode-floor overlay: `_glm53_mixed_prefill_gate` hands down a late-admit cap of
512..1792 tokens, every one of which is < block_size (3584), so a held cold prefill
gets ZERO tokens per step for as long as a peer keeps decoding, instead of crawling
out on the v3 aging ladder. The whole point of that ladder is defeated.

DORMANT IN PRODUCTION: at the production LPTT=1792, `block_size <= max_prefill_tokens`
is 3584 <= 1792 = False, so the modified disjunct is unreachable and this patch is a
provable no-op. It only becomes live in a window that raises LPTT to >= block_size --
which is exactly the window it is a prerequisite for. W34a ran static 3584 without it
and escaped only because no cold prefill was ever co-scheduled with a decoding peer.

FIX: truncate to the block boundary only when that leaves forward progress. When it
does not, allow the chunk to advance sub-block and re-align at the next boundary --
which is precisely what the LPTT < block_size regime (i.e. production today) already
does, so the behaviour is established rather than novel. Mandatory stops are applied
after this point and are unaffected.

GLM53_ALIGN_FLOOR:
  1 / on (default) -- only truncate when aligned_end > start
  0 / off          -- exact upstream behaviour (single-knob rollback, no JIT wipe)

The value is read ONCE and cached in a module global, so changing GLM53_ALIGN_FLOOR
requires a process restart -- it cannot be flipped on a live engine.

Fails closed if the vLLM scheduler anchors drift.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_SCHEDULER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py",
    )
)
MARK = "# [glm53-align-floor]"

IMPORT_OLD = "import itertools\nimport time\n"
IMPORT_NEW = "import itertools\nimport os\nimport time\n"

HELPER = '''
_GLM53_ALIGN_FLOOR = None


def _glm53_align_floor():
    """True when the align-floor fix is enabled (default on).  # [glm53-align-floor]

    Off restores the exact upstream disjunct, so rollback is one env value with
    no kernel-shape change and therefore no JIT cache wipe.
    """
    global _GLM53_ALIGN_FLOOR
    if _GLM53_ALIGN_FLOOR is None:
        v = os.environ.get("GLM53_ALIGN_FLOOR", "1").strip().lower()
        _GLM53_ALIGN_FLOOR = v not in ("0", "off", "false", "no", "")
    return _GLM53_ALIGN_FLOOR

'''

SPLIT_OLD = """            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end
"""

SPLIT_NEW = """            aligned_end = end // block_size * block_size
            # [glm53-align-floor] Truncate to the block boundary only when that
            # leaves forward progress. With LPTT >= block_size the second
            # disjunct fired even when aligned_end <= start, zeroing any
            # sub-block chunk (notably the decode-floor late-admit ladder,
            # whose caps are all < block_size) and stalling the request for as
            # long as a peer decoded. Advancing sub-block and re-aligning at
            # the next boundary is what the LPTT < block_size regime already
            # does. Mandatory stops below are unaffected.
            if aligned_end > start or (
                block_size <= max_prefill_tokens and not _glm53_align_floor()
            ):
                end = aligned_end
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def validate_installed(text: str) -> None:
    """Assert the install is COMPLETE, not merely marked.

    A marker-only idempotence check is fail-open: it reports success on a
    partial install, on a marker left behind after another overlay rewrote the
    condition, and -- worst -- on a syntactically broken file, because the
    early return would skip the AST check entirely. Verify every piece.
    """
    tree = ast.parse(text, filename=str(P))
    if text.count(SPLIT_NEW) != 1:
        raise SystemExit(f"{P}: patched split block not present exactly once")
    if text.count(SPLIT_OLD) != 0:
        raise SystemExit(f"{P}: upstream split block still present after install")
    if text.count(HELPER) != 1:
        raise SystemExit(f"{P}: helper missing, duplicated, or drifted")
    has_plain_os_import = any(
        isinstance(node, ast.Import)
        and any(a.name == "os" and a.asname is None for a in node.names)
        for node in tree.body
    )
    if not has_plain_os_import:
        raise SystemExit(f"{P}: helper requires a top-level plain 'import os'")


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()

    if MARK in text:
        validate_installed(text)
        print(f"{P.name}: {MARK} already present and complete - skipping")
        return 0

    # A partial or colliding state must never be treated as a fresh install.
    if SPLIT_NEW in text or "def _glm53_align_floor(" in text:
        raise SystemExit(f"{P}: partial or conflicting align-floor install detected")

    if "import os\n" not in text.split("import time\n", 1)[0]:
        text = replace_once(text, IMPORT_OLD, IMPORT_NEW, "import os")

    if "def _glm53_align_floor(" not in text:
        needle = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
        if text.count(needle) != 1:
            raise SystemExit(f"{P}: helper insert point not unique")
        text = text.replace(needle, HELPER + needle, 1)

    text = replace_once(text, SPLIT_OLD, SPLIT_NEW, "mamba-align-split")

    validate_installed(text)
    # Atomic replace: a truncated scheduler.py would be unrecoverable mid-boot.
    tmp = P.with_suffix(P.suffix + ".glm53-align-floor.tmp")
    tmp.write_text(text)
    os.replace(tmp, P)
    print(f"patched {P.name} (align floor={os.environ.get('GLM53_ALIGN_FLOOR', '1')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
