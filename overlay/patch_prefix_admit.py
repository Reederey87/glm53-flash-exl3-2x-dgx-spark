#!/usr/bin/env python3
"""Hold a new request while a running partner is still publishing its prefix.

Prefix lookup runs once, when ``num_computed_tokens`` is still 0. A second
copy of a long prompt admitted on that step records whatever pages are
already hashed and then never looks again, so the rest of the shared prefix
is computed twice.

The hold is a scheduler deferral only. It does not pin KV, change the page
size, or call ``get_computed_blocks`` (that lookup can emit block-stored
events for a request that is not admitted). A waiter stays in the skipped
queue until some running request has published the longest full page they
share. A different prefix is not held, and a held request is popped so the
next waiter in the same step still runs.

The grid is ``max(block_size, hash_block_size)``. A finer hash must not
release the waiter before the coarser KV page is hashed.

GLM53_PREFIX_ADMIT=1 applies the edit. Any other value leaves the installed
file untouched. Rollback is the flag at 0 and a new container.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

MARK = "[glm53-prefix-admit]"
SCHEDULER = Path(
    os.environ.get(
        "GLM53_SCHEDULER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py",
    )
)

INSERT_AT = (
    "logger = init_logger(__name__)\n\n\nclass Scheduler(SchedulerInterface):\n"
)

ANCHOR = """\
                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
"""

CALL = """\
                # [glm53-prefix-admit] Lookup below runs once. Wait until a
                # running partner has hashed the longest full page this
                # prompt shares, then admit. Pop so a different prefix
                # behind this waiter is still considered this step.
                if request.num_computed_tokens == 0:
                    _glm53_prefix_need = _glm53_hold_for_inflight_prefix(
                        request,
                        self.running,
                        self.block_size,
                        self.hash_block_size,
                    )
                    if _glm53_prefix_need:
                        if not getattr(
                            request, "_glm53_prefix_admit_logged", False
                        ):
                            logger.info(
                                "[glm53-prefix-admit] holding %s for %s"
                                " published tokens",
                                request.request_id,
                                _glm53_prefix_need,
                            )
                            request._glm53_prefix_admit_logged = True
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue

"""


def _glm53_aligned_prefix_tokens(mine, theirs, block: int) -> int:
    """Block-aligned token LCP. ``block`` is the coarser of the two grids.

    Slices compare in C. A mismatch stops the scan, so a different prompt
    does not walk the whole running prefill.
    """
    if block <= 0:
        return 0
    try:
        limit = min(len(mine), len(theirs))
    except TypeError:
        return 0
    matched = 0
    step = 256
    while matched < limit:
        nxt = matched + step
        if nxt > limit:
            nxt = limit
        try:
            same = mine[matched:nxt] == theirs[matched:nxt]
        except TypeError:
            return (matched // block) * block
        if same:
            matched = nxt
            continue
        while matched < nxt:
            try:
                if mine[matched] != theirs[matched]:
                    return (matched // block) * block
            except TypeError:
                return (matched // block) * block
            matched += 1
        break
    return (matched // block) * block


def _glm53_hold_for_inflight_prefix(request, running, block_size, hash_block_size):
    """Tokens of shared prefix still unpublished, or 0 when the waiter can admit.

    0 means: no running partner shares a full page, or the partner with the
    longest share has already published it. A shorter partner that is ahead
    does not release a longer share that is still in flight.
    """
    try:
        block = max(int(block_size), int(hash_block_size))
    except (TypeError, ValueError):
        return 0
    if block <= 0:
        return 0
    mine = getattr(request, "prompt_token_ids", None)
    if not mine:
        return 0
    shares = getattr(request, "_glm53_prefix_shares", None)
    if shares is None:
        shares = {}
        try:
            request._glm53_prefix_shares = shares
        except Exception:
            shares = {}
    own_id = getattr(request, "request_id", None)
    best = 0
    best_ready = False
    for partner in running:
        if partner is request:
            continue
        partner_id = getattr(partner, "request_id", None)
        if own_id is not None and partner_id == own_id:
            continue
        theirs = getattr(partner, "prompt_token_ids", None)
        if not theirs:
            continue
        shared = shares.get(partner_id) if partner_id is not None else None
        if shared is None:
            shared = _glm53_aligned_prefix_tokens(mine, theirs, block)
            if partner_id is not None:
                shares[partner_id] = shared
        if shared <= 0:
            continue
        try:
            computed = int(getattr(partner, "num_computed_tokens", 0) or 0)
        except (TypeError, ValueError):
            computed = 0
        if computed < 0:
            computed = 0
        published = (computed // block) * block
        ready = published >= shared
        if shared > best:
            best = shared
            best_ready = ready
        elif shared == best and ready:
            best_ready = True
    if best > 0 and not best_ready:
        return best
    return 0


def _helper_source() -> str:
    parts = (
        inspect.getsource(_glm53_aligned_prefix_tokens),
        inspect.getsource(_glm53_hold_for_inflight_prefix),
    )
    return "\n\n".join(parts) + "\n\n\n"


def _write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".glm53-tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _complete(text: str) -> bool:
    return (
        text.count(MARK) == 2
        and text.count("def _glm53_aligned_prefix_tokens(") == 1
        and text.count("def _glm53_hold_for_inflight_prefix(") == 1
        and text.count(CALL) == 1
        and text.count(ANCHOR) == 1
    )


def main() -> int:
    flag = os.environ.get("GLM53_PREFIX_ADMIT", "0").strip()
    if flag == "0":
        print("GLM53_PREFIX_ADMIT is off — installed sources unchanged", flush=True)
        return 0
    if flag != "1":
        print(
            f"GLM53_PREFIX_ADMIT must be exactly 0 or 1 (got: {flag!r})",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if not SCHEDULER.is_file():
        print(f"{MARK} missing {SCHEDULER}", file=sys.stderr, flush=True)
        return 1
    text = SCHEDULER.read_text()
    if _complete(text):
        print(f"{MARK} already present in {SCHEDULER}", flush=True)
        return 0
    if MARK in text or "def _glm53_hold_for_inflight_prefix(" in text:
        print(f"{MARK} partial install in {SCHEDULER}", file=sys.stderr, flush=True)
        return 1
    if text.count(ANCHOR) != 1 or text.count(INSERT_AT) != 1:
        print(
            f"{MARK} anchor drift in {SCHEDULER}: "
            f"lookup={text.count(ANCHOR)} class={text.count(INSERT_AT)}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    helper = "logger = init_logger(__name__)\n\n\n" + _helper_source()
    helper += "class Scheduler(SchedulerInterface):\n"
    patched = text.replace(INSERT_AT, helper, 1)
    patched = patched.replace(ANCHOR, CALL + ANCHOR, 1)
    if not _complete(patched):
        print(f"{MARK} patch did not land cleanly in {SCHEDULER}", file=sys.stderr, flush=True)
        return 1
    compile(patched, str(SCHEDULER), "exec")
    _write(SCHEDULER, patched)
    print(f"{MARK} applied to {SCHEDULER}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
