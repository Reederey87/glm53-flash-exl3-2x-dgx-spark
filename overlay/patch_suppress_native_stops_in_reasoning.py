#!/usr/bin/env python3
"""Keep native stop-token ids dormant until reasoning ends (glm53-flash).

Companion to ``patch_suppress_stops_in_reasoning.py``, which dormant-izes
*client-provided stop strings* while ``<think>`` is open but explicitly
leaves native EOS/stop-token handling untouched. A native GLM control token
(e.g. ``<|observation|>``, the model's own tool-turn boundary marker) can
also land in ``sampling_params.stop_token_ids`` and fire mid-reasoning,
finishing the request with ``stop_reason=<token_id>`` and empty visible
content/tool_calls before the model ever closes ``<think>``.

Anchor is this image's ``v1/core/sched/utils.py::check_stop``. Reasoning-open
is checked two ways, either one is enough to dormant-ize the stop:

1. vLLM's own reasoning-parser state
   (``request.structured_output_request.reasoning_ended is False``). Cheap,
   precise, but only populated when the request actually has structured-
   output constraints active (``StructuredOutputRequest.from_sampling_params``
   returns ``None`` otherwise) — a plain ``tools`` + ``tool_choice: "auto"``
   call with no forced JSON schema never populates it. Confirmed in
   production (2026-09-01/09-02): 11/11 empty-completion hits on one scan
   were exactly this uncovered shape.
2. Independent of structured-output state entirely: this deployment's chat
   template primes every thinking-enabled request with the prompt ending in
   the ``<think>`` token (think-in-prompt). If the prompt ends there and
   ``</think>`` hasn't appeared yet in ``request.output_token_ids``,
   reasoning is still open — regardless of whether any grammar/structured-
   output tracking ever ran for this request. This is what actually covers
   the auto-tool_choice gap above.

v2 (2026-09-02) treated an unambiguous native signal as authoritative and
skipped the token check whenever ``reasoning_ended`` was ``True`` or
``False`` (not just ``None``). Reverted: after deploying it, a live scan
still climbed 3/12 -> 10/12 empty-completion hits over ~12 minutes,
correlating with heavy concurrent load. ``patch_xgrammar_termination.py``'s
own docstring implies structured-output/grammar tracking likely *is*
active for this deployment's auto-tool-choice calls (contrary to the
"never populated" claim above, which was not independently verified) — if
so, a ``reasoning_ended`` flip to ``True`` under load-induced timing is a
more plausible live failure mode than the token check ever incorrectly
overriding a confirmed-closed native signal (which was the theoretical
risk the v2 change guarded against, never observed). Back to plain OR:
either signal saying "still open" is enough to dormant-ize the stop.

``eos_token_id`` and length caps are unchanged, matching the stop-string
patch's own invariant. Token ids for ``<think>``/``</think>`` are pinned to
this model's tokenizer (``brandonmusic/GLM-5.3-Flash-tr3-4bpw``); override
via ``GLM53_THINK_START_TOKEN_ID`` / ``GLM53_THINK_END_TOKEN_ID`` if the
model changes.

Opt-out: ``GLM53_SUPPRESS_STOPS_IN_REASONING=0`` or
``VLLM_SUPPRESS_STOPS_IN_REASONING=0`` (same switch as the stop-string
patch, since this is the same feature covering the other stop source).

Every time this guard actually engages (i.e. a native stop-token candidate
was about to fire), it logs one INFO line tagged
``[suppress-native-stops-in-reasoning]`` with both raw signal values and
the final decision — deliberately not gated behind
``VLLM_LOGGING_LEVEL=DEBUG`` (that crashed this box's CUDA-graph warmup
twice in this same session; this uses the engine's always-on INFO logger
instead) so the next occurrence is directly diagnosable from the normal
server log instead of inferred from client-side timestamps.
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

LOGGER_IMPORT_OLD = "from vllm.sampling_params import RepetitionDetectionParams\n"
LOGGER_IMPORT_NEW = (
    "from vllm.logger import init_logger\n"
    "from vllm.sampling_params import RepetitionDetectionParams\n"
)
LOGGER_INIT_OLD = "from vllm.v1.request import Request, RequestStatus\n"
LOGGER_INIT_NEW = (
    "from vllm.v1.request import Request, RequestStatus\n\n"
    "_native_stop_logger = init_logger(__name__)\n"
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
        # Either of two independent signals being "still open" is enough:
        # vLLM's own reasoning-parser state (when structured-output tracking
        # is active for this request), or an independent token-based check
        # (prompt ends in <think>, </think> not seen yet in the output) -
        # covers requests where the native signal is unavailable/unknown, or
        # (2026-09-02) possibly wrong under load. Logged every time this
        # fires - see module docstring.
        if not _native_stop_reasoning_open(request):
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


def _reasoning_open_by_tokens(request) -> bool:
    # [suppress-native-stops-in-reasoning] structured-output-agnostic check:
    # this deployment's chat template primes every thinking-enabled request
    # with the prompt ending in <think> (think-in-prompt). If so, and
    # </think> hasn't appeared yet in the output, reasoning is still open
    # regardless of whether any grammar/structured-output tracking ran.
    # Fail closed (treat as "not open") on any unexpected error - a bad
    # env var or an unforeseen edge case here must never crash check_stop,
    # only fall back to pre-patch behavior for that one request.
    try:
        think_start = int(os.environ.get("GLM53_THINK_START_TOKEN_ID", "154841"))
        think_end = int(os.environ.get("GLM53_THINK_END_TOKEN_ID", "154842"))
        ptids = request.prompt_token_ids
        if not ptids or ptids[-1] != think_start:
            return False
        return think_end not in request.output_token_ids
    except Exception:
        return False


def _native_stop_reasoning_open(request) -> bool:
    # [suppress-native-stops-in-reasoning] single entry point combining both
    # signals: plain OR, either one saying "still open" is enough. Each
    # signal is separately fail-closed - if the structured-output check
    # errors, that alone must not take down the token-based check too, so
    # it's isolated to its own try/except rather than one that wraps (and
    # could short-circuit) both signals. Logs every time this guard is
    # consulted for an actual stop-token candidate, tagged for grep, so a
    # future occurrence is directly diagnosable from the server log.
    if not _suppress_native_stops_enabled():
        return False
    sor = request.structured_output_request
    native_reasoning_ended = None
    native_open = False
    try:
        if sor is not None:
            native_reasoning_ended = sor.reasoning_ended
            native_open = sor.reasoning_ended is False
    except Exception:
        pass
    token_open = _reasoning_open_by_tokens(request)
    result = native_open or token_open
    try:
        _native_stop_logger.info(
            "[suppress-native-stops-in-reasoning] request_id=%s "
            "sor_present=%s reasoning_ended=%s token_check_open=%s -> %s",
            getattr(request, "request_id", "?"),
            sor is not None,
            native_reasoning_ended,
            token_open,
            "SUPPRESSED (still open)" if result else "STOP FIRES",
        )
    except Exception:
        pass
    return result
'''


def apply_text(src: str) -> tuple[str, str]:
    """Return (new_source, status): applied|skipped|missing:..."""
    if MARK in src and "_reasoning_open_by_tokens" in src:
        return src, "skipped"
    missing = []
    if IMPORT_OLD not in src:
        missing.append("import")
    if LOGGER_IMPORT_OLD not in src:
        missing.append("logger_import")
    if LOGGER_INIT_OLD not in src:
        missing.append("logger_init")
    if CHECK_OLD not in src:
        missing.append("check_stop")
    if missing:
        return src, "missing:" + ",".join(missing)
    out = src.replace(IMPORT_OLD, IMPORT_NEW, 1)
    out = out.replace(LOGGER_IMPORT_OLD, LOGGER_IMPORT_NEW, 1)
    out = out.replace(LOGGER_INIT_OLD, LOGGER_INIT_NEW, 1)
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
