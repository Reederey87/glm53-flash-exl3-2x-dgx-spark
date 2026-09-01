#!/usr/bin/env python3
"""Keep native stop-token ids dormant until reasoning ends (glm53-flash).

Companion to ``patch_suppress_stops_in_reasoning.py``, which dormant-izes
*client-provided stop strings* while ``<think>`` is open but explicitly
leaves native EOS/stop-token handling untouched. A native GLM control token
(e.g. ``<|observation|>``, the model's own tool-turn boundary marker) can
also land in ``sampling_params.stop_token_ids`` and fire mid-reasoning,
finishing the request with ``stop_reason=<token_id>`` and empty visible
content/tool_calls before the model ever closes ``<think>``.

Anchor is this image's ``v1/core/sched/utils.py::check_stop``. Unlike the
stop-string patch, this reuses vLLM's own reasoning-parser state
(``request.structured_output_request.reasoning_ended``) instead of
re-deriving it from token ids — that field is only positively ``False``
(vs. ``None``/unset) once a reasoning parser is active for the request, so
this only ever engages for requests that actually have one configured
(guided decoding / tool-call grammar), which is exactly the affected
workload. ``eos_token_id`` and length caps are unchanged, matching the
stop-string patch's own invariant.

Opt-out: ``GLM53_SUPPRESS_STOPS_IN_REASONING=0`` or
``VLLM_SUPPRESS_STOPS_IN_REASONING=0`` (same switch as the stop-string
patch, since this is the same feature covering the other stop source).
"""
from __future__ import annotations

import sys
from pathlib import Path

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/utils.py")
MARK = "# [suppress-native-stops-in-reasoning]"

IMPORT_OLD = (
    "import contextlib\n"
    "from collections.abc import Sequence\n"
)
IMPORT_NEW = (
    "import contextlib\n"
    "import os\n"
    "from collections.abc import Sequence\n"
)

CHECK_OLD = """    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
"""

CHECK_NEW = """    if last_token_id in (sampling_params.stop_token_ids or ()):
        # [suppress-native-stops-in-reasoning] mirror
        # patch_suppress_stops_in_reasoning.py for native stop-token ids:
        # a control token like <|observation|> can land in stop_token_ids
        # and fire mid-<think>, finishing the turn with no visible output.
        # Only engages once a reasoning parser has positively confirmed
        # we're still open (reasoning_ended is False, not None/unset).
        sor = request.structured_output_request
        reasoning_open = (
            _suppress_native_stops_enabled()
            and sor is not None
            and sor.reasoning_ended is False
        )
        if not reasoning_open:
            request.status = RequestStatus.FINISHED_STOPPED
            request.stop_reason = last_token_id
            return True
"""

HELPER = '''

def _suppress_native_stops_enabled() -> bool:
    # [suppress-native-stops-in-reasoning]
    for key in (
        "GLM53_SUPPRESS_STOPS_IN_REASONING",
        "VLLM_SUPPRESS_STOPS_IN_REASONING",
    ):
        if key in os.environ:
            return os.environ.get(key) != "0"
    return True
'''


def apply_text(src: str) -> tuple[str, str]:
    """Return (new_source, status): applied|skipped|missing:..."""
    if MARK in src and "_suppress_native_stops_enabled" in src:
        return src, "skipped"
    missing = []
    if IMPORT_OLD not in src:
        missing.append("import")
    if CHECK_OLD not in src:
        missing.append("check_stop")
    if missing:
        return src, "missing:" + ",".join(missing)
    out = src.replace(IMPORT_OLD, IMPORT_NEW, 1)
    out = out.replace(CHECK_OLD, CHECK_NEW, 1)
    out = out.rstrip("\n") + "\n\n\n" + HELPER.strip("\n") + "\n"
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
        print("suppress-native-stops-in-reasoning:", "APPLIED" if applied else "NOT APPLIED")
        return 0
    target = Path(argv[1]) if len(argv) > 1 else P
    if not target.is_file():
        print(f"[suppress-native-stops-in-reasoning] missing {target}", file=sys.stderr)
        return 1
    status = apply_file(target)
    print(f"[suppress-native-stops-in-reasoning] {status}: {target}")
    return 0 if status.startswith("applied") or status == "skipped" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
