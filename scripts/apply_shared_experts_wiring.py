#!/usr/bin/env python3
"""Apply the GLM53_SHARED_EXPERTS_EARLY wiring to a runtime start.sh.

The runtime tree's ``start.sh`` is a different lineage from the repo's: it lacks
several repo-only knobs whose host files are absent there, so copying the repo
file wholesale would make the launcher ``die`` on a missing patch host. This
script applies only the shared-experts delta, by exact anchor, and refuses on a
missing or ambiguous anchor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

VAR_BLOCK = """SHARED_EXPERTS_EARLY_PATCH_HOST="${SHARED_EXPERTS_EARLY_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_shared_experts_overlap.py}"
"""

# The strict-bool loop validates every knob it names, so every one of those
# knobs needs a concrete value assigned by the W41/W42 defaults block, which the
# numeric-config test harness slices in ahead of validate_numeric_config. The
# unset-only form keeps "" a value (and therefore a rejected one).
DEFAULTS_BLOCK = """# LOCAL: shared-expert overlap. 1 = enqueue the MoE shared experts on the
# auxiliary CUDA stream at the sync point, before the gate and the routed
# dispatch, instead of in forward() after the routed experts. Python-only; it
# reorders work, it does not change the arithmetic. Unset-only default so ""
# stays a value and is rejected, like the knobs above.
GLM53_SHARED_EXPERTS_EARLY="${GLM53_SHARED_EXPERTS_EARLY-0}"
"""

VALIDATOR = """    case "${GLM53_SHARED_EXPERTS_EARLY-0}" in
        0|1) ;;
        *) echo "GLM53_SHARED_EXPERTS_EARLY must be exactly 0 or 1 (got: '${GLM53_SHARED_EXPERTS_EARLY}')" >&2; return 2 ;;
    esac
"""

APPLY = """# Self-gated on GLM53_SHARED_EXPERTS_EARLY. Flag off does not edit
# shared_experts.py.
if [ -f /opt/glm53/patch_shared_experts_overlap.py ]; then
    python3 -S /opt/glm53/patch_shared_experts_overlap.py
fi
"""

SCP = """    [ -f "$SHARED_EXPERTS_EARLY_PATCH_HOST" ] || die "missing $SHARED_EXPERTS_EARLY_PATCH_HOST"
    scp -q -o BatchMode=yes "$SHARED_EXPERTS_EARLY_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_shared_experts_overlap.py"
"""

# (anchor, replacement, expected occurrence count)
EDITS: list[tuple[str, str, int]] = [
    (
        '_glm53_cli_ptd_set="${GLM53_PROMPT_TOKENS_DETAILS+a}"\n'
        '_glm53_cli_ptd_val="${GLM53_PROMPT_TOKENS_DETAILS-}"\n',
        '_glm53_cli_ptd_set="${GLM53_PROMPT_TOKENS_DETAILS+a}"\n'
        '_glm53_cli_ptd_val="${GLM53_PROMPT_TOKENS_DETAILS-}"\n'
        '_glm53_cli_sxe_set="${GLM53_SHARED_EXPERTS_EARLY+a}"\n'
        '_glm53_cli_sxe_val="${GLM53_SHARED_EXPERTS_EARLY-}"\n',
        1,
    ),
    (
        '[ -n "${_glm53_cli_ptd_set}" ] && GLM53_PROMPT_TOKENS_DETAILS="$_glm53_cli_ptd_val"\n',
        '[ -n "${_glm53_cli_ptd_set}" ] && GLM53_PROMPT_TOKENS_DETAILS="$_glm53_cli_ptd_val"\n'
        '[ -n "${_glm53_cli_sxe_set}" ] && GLM53_SHARED_EXPERTS_EARLY="$_glm53_cli_sxe_val"\n',
        1,
    ),
    (
        'GLM53_ROUTER_ONCE="${GLM53_ROUTER_ONCE-0}"\n'
        'ROUTER_ONCE_PATCH_HOST="${ROUTER_ONCE_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_router_once.py}"\n',
        'GLM53_ROUTER_ONCE="${GLM53_ROUTER_ONCE-0}"\n'
        'ROUTER_ONCE_PATCH_HOST="${ROUTER_ONCE_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_router_once.py}"\n'
        + VAR_BLOCK,
        1,
    ),
    (
        "# LOCAL: W41/W42 knob defaults (end)\n",
        DEFAULTS_BLOCK + "# LOCAL: W41/W42 knob defaults (end)\n",
        1,
    ),
    (
        '    case "${GLM53_ROUTER_ONCE-0}" in\n'
        "        0|1) ;;\n"
        "        *) echo \"GLM53_ROUTER_ONCE must be exactly 0 or 1 (got: '${GLM53_ROUTER_ONCE}')\" >&2; return 2 ;;\n"
        "    esac\n",
        '    case "${GLM53_ROUTER_ONCE-0}" in\n'
        "        0|1) ;;\n"
        "        *) echo \"GLM53_ROUTER_ONCE must be exactly 0 or 1 (got: '${GLM53_ROUTER_ONCE}')\" >&2; return 2 ;;\n"
        "    esac\n" + VALIDATOR,
        1,
    ),
    (
        "    for _v in GLM53_KV_CAPACITY_LOG GLM53_APC_NO_STORE GLM53_PROMPT_TOKENS_DETAILS "
        "GLM53_EXL3_MOE_PIPELINE GLM53_EXL3_MOE_REUSE; do\n",
        "    for _v in GLM53_KV_CAPACITY_LOG GLM53_APC_NO_STORE GLM53_PROMPT_TOKENS_DETAILS "
        "GLM53_EXL3_MOE_PIPELINE GLM53_EXL3_MOE_REUSE GLM53_SHARED_EXPERTS_EARLY; do\n",
        1,
    ),
    (
        "# Self-gated; flag off leaves model.py unchanged.\n"
        "if [ -f /opt/glm53/patch_router_once.py ]; then\n"
        "    python3 -S /opt/glm53/patch_router_once.py\n"
        "fi\n",
        "# Self-gated; flag off leaves model.py unchanged.\n"
        "if [ -f /opt/glm53/patch_router_once.py ]; then\n"
        "    python3 -S /opt/glm53/patch_router_once.py\n"
        "fi\n" + APPLY,
        2,
    ),
    (
        '    [ -f "$ROUTER_ONCE_PATCH_HOST" ] || die "missing $ROUTER_ONCE_PATCH_HOST"\n'
        '    scp -q -o BatchMode=yes "$ROUTER_ONCE_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_router_once.py"\n',
        '    [ -f "$ROUTER_ONCE_PATCH_HOST" ] || die "missing $ROUTER_ONCE_PATCH_HOST"\n'
        '    scp -q -o BatchMode=yes "$ROUTER_ONCE_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_router_once.py"\n'
        + SCP,
        1,
    ),
    (
        '        -e "GLM53_ROUTER_ONCE=$GLM53_ROUTER_ONCE"\n',
        '        -e "GLM53_ROUTER_ONCE=$GLM53_ROUTER_ONCE"\n'
        '        -e "GLM53_SHARED_EXPERTS_EARLY=$GLM53_SHARED_EXPERTS_EARLY"\n',
        1,
    ),
    (
        "             GLM53_PROFILE_TORCH_DIR GLM53_PROFILE_MAX_ITERS \\\n"
        "             GLM53_PROMPT_TOKENS_DETAILS; do\n",
        "             GLM53_PROFILE_TORCH_DIR GLM53_PROFILE_MAX_ITERS \\\n"
        "             GLM53_SHARED_EXPERTS_EARLY \\\n"
        "             GLM53_PROMPT_TOKENS_DETAILS; do\n",
        1,
    ),
    (
        "        -v '/tmp/patch_router_once.py:/opt/glm53/patch_router_once.py:ro' \\\n",
        "        -v '/tmp/patch_router_once.py:/opt/glm53/patch_router_once.py:ro' \\\n"
        "        -v '/tmp/patch_shared_experts_overlap.py:/opt/glm53/patch_shared_experts_overlap.py:ro' \\\n",
        1,
    ),
    (
        '        -v "$ROUTER_ONCE_PATCH_HOST:/opt/glm53/patch_router_once.py:ro" \\\n',
        '        -v "$ROUTER_ONCE_PATCH_HOST:/opt/glm53/patch_router_once.py:ro" \\\n'
        '        -v "$SHARED_EXPERTS_EARLY_PATCH_HOST:/opt/glm53/patch_shared_experts_overlap.py:ro" \\\n',
        1,
    ),
]

MARKER = "GLM53_SHARED_EXPERTS_EARLY"


def apply(text: str) -> tuple[str, int]:
    total = 0
    for anchor, replacement, count in EDITS:
        found = text.count(anchor)
        if found != count:
            raise SystemExit(
                f"anchor matched {found}x, expected {count}x: {anchor.splitlines()[0]!r}"
            )
        text = text.replace(anchor, replacement)
        total += count
    return text, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out")
    args = ap.parse_args()
    src = Path(args.src)
    original = src.read_text()
    if MARKER in original:
        print(f"{src}: already wired (marker present) — no change")
        return 0
    updated, total = apply(original)
    if args.out:
        Path(args.out).write_text(updated)
        print(f"wrote {args.out} ({total} edits)")
    else:
        sys.stdout.write(updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
