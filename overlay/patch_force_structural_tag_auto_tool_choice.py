#!/usr/bin/env python3
"""Opt-in: build the tool-call structural tag for plain tool_choice="auto" too (glm53-flash).

**Prepared, not enabled.** Default-off. Untested hypothesis, no controlled
A/B yet — do not wire into ``start.sh`` or restart the server on this alone.
See ``docs/05-known-issues.md`` section 5 for the observation that motivated
this: on a live scan, a `tool_choice: "auto"` request emptied out
(`<|observation|>` firing with no visible content/tool_calls) *after*
reasoning had already genuinely closed
(`sor_present=False reasoning_ended=None token_check_open=False` in
``patch_suppress_native_stops_in_reasoning.py``'s diagnostic log). That is
not the mid-`<think>` bug that patch fixes — it is the model choosing to
emit nothing before the tool-turn boundary token, which no stop-token patch
can address.

Root cause traced to ``vllm/tool_parsers/structural_tag_registry.py::get_model_structural_tag``:

    if tool_choice == "auto" and not _any_tool_strict(tools):
        return None

Upstream only builds the xgrammar structural tag (grammar-constrained
decoding forcing a valid tool-call-or-text shape) for `tool_choice: "auto"`
when at least one tool in the request sets `strict: true` — mirroring
OpenAI's strict-function-calling opt-in. A plain `tools` + `tool_choice:
"auto"` call with no `strict` flags (this deployment's third-party agentic
client's exact shape) never gets a structural tag, so:

1. ``request.structured_outputs`` stays ``None``, so
   ``request.structured_output_request`` stays ``None`` too — signal (1) in
   ``patch_suppress_native_stops_in_reasoning.py`` (``reasoning_ended``) can
   never populate for this shape, regardless of that patch. (Signal (2),
   the token-based check, already covers the *mid-reasoning* case
   independently of this — this patch is about the *post-reasoning* empty
   output case that signal (2) cannot help with either, since token_check_open
   is correctly False once ``</think>`` has really been emitted.)
2. Decoding is fully unconstrained for these requests — nothing stops the
   model from sampling a control/boundary token with no preceding visible
   payload.

**Hypothesis (unverified):** forcing the structural tag on for plain-auto
requests too would (a) give signal (1) real coverage for this request
shape, and (b) may reduce the empty-completion rate itself, since
grammar-constrained decoding forces output into a valid tool-call-or-text
shape at every step rather than leaving the model free to emit a bare
control token. Neither claim is verified against this deployment — needs a
controlled A/B (``scripts/bench.py``-style matched-prompt comparison, or a
repro script hitting the endpoint with/without this patch enabled) before
being treated as a real fix, not just a plausible one. Enabling this also
changes decoding behavior for every plain-auto tool-calling request, not
just the empty-completion path — a real behavior change, not a no-op
guard like the stop-token patches.

Opt-in only, default OFF: ``GLM53_FORCE_AUTO_TOOL_STRUCTURAL_TAG=1``. With
the var unset or ``0``, this patch is a no-op even once applied — the
original upstream gate is preserved byte-for-byte, just wrapped in an
additional condition that is False by default.

Anchor: ``vllm/tool_parsers/structural_tag_registry.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

P = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/structural_tag_registry.py"
)
MARK = "# [force-structural-tag-auto-tool-choice]"

IMPORT_OLD = "from collections.abc import Callable, Sequence\n"
IMPORT_NEW = "import os\nfrom collections.abc import Callable, Sequence\n"

GATE_OLD = """    if not tools or tool_choice == "none":
        return None

    if tool_choice == "auto" and not _any_tool_strict(tools):
        return None
"""

GATE_NEW = """    if not tools or tool_choice == "none":
        return None

    if (
        tool_choice == "auto"
        and not _any_tool_strict(tools)
        and not _force_auto_tool_structural_tag_enabled()
    ):
        # [force-structural-tag-auto-tool-choice]
        return None
"""

HELPER = '''

def _force_auto_tool_structural_tag_enabled() -> bool:
    # [force-structural-tag-auto-tool-choice] opt-in, default OFF - see
    # module docstring in patch_force_structural_tag_auto_tool_choice.py.
    # Untested hypothesis, no controlled A/B yet: do not flip this on in
    # production without one.
    return os.environ.get("GLM53_FORCE_AUTO_TOOL_STRUCTURAL_TAG", "0") == "1"
'''


def apply_text(src: str) -> tuple[str, str]:
    """Return (new_source, status): applied|skipped|missing:..."""
    if MARK in src and "_force_auto_tool_structural_tag_enabled" in src:
        return src, "skipped"
    missing = []
    if IMPORT_OLD not in src:
        missing.append("import")
    if GATE_OLD not in src:
        missing.append("gate")
    if missing:
        return src, "missing:" + ",".join(missing)
    out = src.replace(IMPORT_OLD, IMPORT_NEW, 1)
    out = out.replace(GATE_OLD, GATE_NEW, 1)
    out = out.rstrip("\n") + "\n" + HELPER.strip("\n") + "\n"
    return out, "applied"


def apply_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    new, status = apply_text(text)
    if status == "applied":
        path.write_text(new, encoding="utf-8")
    return status


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "--status":
        target = Path(argv[2]) if len(argv) > 2 else P
        applied = target.is_file() and MARK in target.read_text()
        print(
            "force-structural-tag-auto-tool-choice:",
            "APPLIED" if applied else "NOT APPLIED",
        )
        return 0
    target = Path(argv[1]) if len(argv) > 1 else P
    if not target.is_file():
        print(f"[force-structural-tag-auto-tool-choice] missing {target}", file=sys.stderr)
        return 1
    status = apply_file(target)
    print(f"[force-structural-tag-auto-tool-choice] {status}: {target}")
    return 0 if status.startswith("applied") or status == "skipped" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
