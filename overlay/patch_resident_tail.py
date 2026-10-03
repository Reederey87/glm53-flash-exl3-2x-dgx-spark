#!/usr/bin/env python3
"""Keep the token budget on a short cached tail until that tail is done.

A cached follow-up that arrives while a cold prefill is already running
waits out that prefill. The running loop is first, and the long-prefill
cap then splits the next steps between the cold chunk and the tail, so
the tail pays the cold chunk's step time.

This overlay does two things, and only for a resident tail: a request
whose cache hit is at least one page and whose uncached remainder fits
in one batched-token budget. While such a tail is running, or is waiting
with a free sequence slot, a cold prefill does not take this step. The
long-prefill cap is also not applied to that tail, so one exclusive step
can finish it. Two cold prefills still share. A decode is never deferred.
The last chunk of a cold prefill is not a tail: the mark is written only
at fresh admission, and cleared when that admission does not qualify.

The waiting-queue check uses ``find_longest_cache_hit``. It does not
allocate blocks. A waiter that an in-flight prefix hold is still keeping
does not count, and a hold that raises is treated as still blocked, so
the publisher of an unpublished page is not deferred for that waiter.
Each cold request is deferred at most a few steps.

No KV pin and no page-size change. A general second-prefill budget is a
different policy (upstream discussion on inter-prefill scheduling); this
edit does not install one.

GLM53_RESIDENT_TAIL=1 applies the edit. Any other value leaves the
installed file untouched. Rollback is the flag at 0 and a new container.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

MARK = "[glm53-resident-tail]"
SCHEDULER = Path(
    os.environ.get(
        "GLM53_SCHEDULER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py",
    )
)

CLASS_LINE = "class Scheduler(SchedulerInterface):\n"

RUN_ANCHOR = (
    "            request = self.running[req_index]\n"
    "            if input_budget <= draft_slots:\n"
    "                break\n"
)

RUN_CALL = """\
            # [glm53-resident-tail] A cold prefill does not share this step
            # with a resident tail. The tail keeps the token budget until
            # its prefill is done, or until the step cap.
            if _glm53_defer_cold_for_resident_tail(self, request):
                req_index += 1
                continue

"""

RUN_LPTT = (
    "            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:\n"
    "                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n"
    "            num_new_tokens = min(\n"
    "                num_new_tokens, token_budget, input_budget - draft_slots\n"
    "            )\n"
)

RUN_LPTT_NEW = (
    "            if (\n"
    "                0\n"
    "                < self.scheduler_config.long_prefill_token_threshold\n"
    "                < num_new_tokens\n"
    "                and not _glm53_tail_still_open(\n"
    "                    request, request.num_computed_tokens\n"
    "                )\n"
    "            ):\n"
    "                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n"
    "            num_new_tokens = min(\n"
    "                num_new_tokens, token_budget, input_budget - draft_slots\n"
    "            )\n"
)

WAIT_LPTT = (
    "                    threshold = self.scheduler_config.long_prefill_token_threshold\n"
    "                    if 0 < threshold < num_new_tokens:\n"
    "                        num_new_tokens = threshold\n"
)

WAIT_LPTT_NEW = (
    "                    _glm53_note_resident_tail(request, num_computed_tokens)\n"
    "                    threshold = self.scheduler_config.long_prefill_token_threshold\n"
    "                    if (\n"
    "                        0 < threshold < num_new_tokens\n"
    "                        and not _glm53_tail_still_open(\n"
    "                            request, num_computed_tokens\n"
    "                        )\n"
    "                    ):\n"
    "                        num_new_tokens = threshold\n"
)


def _glm53_resident_tail_limit(name: str, default: int, lo: int, hi: int) -> int:
    """Inclusive bound. A missing or unusable value falls back to ``default``.

    ``os`` is imported here because the installed scheduler does not have to
    import it. The function is copied into that file as text.
    """
    import os

    raw = os.environ.get(name, str(default))
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    if value < lo or value > hi:
        return default
    return value


def _glm53_resident_tail_page() -> int:
    """Cache-hit floor, in tokens. One MLA page on this deployment."""
    return _glm53_resident_tail_limit("GLM53_RESIDENT_TAIL_PAGE", 3584, 64, 65536)


def _glm53_resident_tail_tokens() -> int:
    """Largest uncached remainder that still counts as a resident tail.

    Matches the mixed-prefill warm bypass, so a tail that keeps the budget
    is also a tail the decode floor will share a step with.
    """
    return _glm53_resident_tail_limit("GLM53_RESIDENT_TAIL_TOKENS", 3584, 1, 65536)


def _glm53_resident_tail_steps() -> int:
    """How many steps one cold prefill may yield before it runs again."""
    return _glm53_resident_tail_limit("GLM53_RESIDENT_TAIL_STEPS", 4, 1, 32)


def _glm53_is_resident_tail(computed, prompt, page, tail) -> bool:
    """True when ``computed`` is a full page and the remainder fits in ``tail``."""
    try:
        computed_i = int(computed)
        prompt_i = int(prompt)
        page_i = int(page)
        tail_i = int(tail)
    except (TypeError, ValueError):
        return False
    if page_i <= 0 or tail_i <= 0:
        return False
    left = prompt_i - computed_i
    return computed_i >= page_i and 0 < left <= tail_i


def _glm53_should_defer_cold(
    *,
    prefill_left: int,
    is_tail: bool,
    tail_waiting: bool,
    tail_running: bool,
    num_running: int,
    max_running: int,
    defer_count: int,
    step_cap: int,
) -> bool:
    """True when this running cold prefill should leave the step to a tail.

    A running tail needs no free slot: it is already admitted. A waiting
    tail does, because deferring tokens does not free a sequence. Decodes
    and a prefill that is itself the tail are never deferred.
    """
    try:
        left = int(prefill_left)
        running = int(num_running)
        limit = int(max_running)
        used = int(defer_count)
        cap = int(step_cap)
    except (TypeError, ValueError):
        return False
    if is_tail or left <= 64 or used >= cap or cap <= 0:
        return False
    if tail_running:
        return True
    if tail_waiting and running < limit:
        return True
    return False


def _glm53_note_resident_tail(request, computed) -> None:
    """Stamp a fresh admission. A request that already has computed tokens is left alone.

    ``computed`` is the local cache hit, which on this path is not yet
    copied onto ``request.num_computed_tokens``. A cold prefill's later
    chunks therefore cannot grow into a tail. A stale stamp is cleared
    when this admission does not qualify, so preemption cannot keep it.
    """
    try:
        already = int(getattr(request, "num_computed_tokens", 0) or 0)
    except (TypeError, ValueError):
        return
    if already != 0:
        return
    try:
        prompt = int(request.num_prompt_tokens)
    except (TypeError, ValueError, AttributeError):
        return
    qualifies = _glm53_is_resident_tail(
        computed,
        prompt,
        _glm53_resident_tail_page(),
        _glm53_resident_tail_tokens(),
    )
    try:
        request._glm53_resident_tail = 1 if qualifies else 0
    except Exception:
        return


def _glm53_tail_still_open(request, computed) -> bool:
    """The admission stamp still describes a short cached remainder."""
    try:
        stamp = int(getattr(request, "_glm53_resident_tail", 0) or 0)
    except (TypeError, ValueError):
        return False
    if stamp != 1:
        return False
    try:
        prompt = int(request.num_prompt_tokens)
    except (TypeError, ValueError, AttributeError):
        return False
    return _glm53_is_resident_tail(
        computed,
        prompt,
        _glm53_resident_tail_page(),
        _glm53_resident_tail_tokens(),
    )


def _glm53_waiter_cache_hit(waiter, manager) -> int:
    """Resident tokens for a not-yet-computed waiter. Negative when it cannot qualify."""
    try:
        computed = int(getattr(waiter, "num_computed_tokens", 0) or 0)
    except (TypeError, ValueError):
        return -1
    if computed != 0:
        return -1
    hashes = getattr(waiter, "block_hashes", None)
    if not hashes:
        return -1
    try:
        ntok = int(getattr(waiter, "num_tokens", 0) or 0)
    except (TypeError, ValueError):
        ntok = 0
    if ntok <= 1:
        try:
            ntok = int(getattr(waiter, "num_prompt_tokens", 0) or 0)
        except (TypeError, ValueError):
            return -1
    if ntok <= 1:
        return -1
    try:
        _blocks, hit, _uncached = manager.coordinator.find_longest_cache_hit(
            hashes, ntok - 1
        )
        return max(0, int(hit))
    except Exception:
        return -1


def _glm53_waiter_is_blocked(hold, waiter, running, block_size, hash_block_size) -> bool:
    """True when the in-flight prefix hold still wants this waiter to wait.

    A missing hold means nothing is blocking the waiter. An exception is
    fail-closed: the publisher of an unpublished page must not be deferred
    because the hold could not be read.
    """
    if hold is None:
        return False
    try:
        return bool(hold(waiter, running, block_size, hash_block_size))
    except Exception:
        return True


def _glm53_has_waiting_resident_tail(scheduler, page, tail) -> bool:
    """True when one of the first four fresh waiters is an unblocked resident tail."""
    try:
        block_size = int(scheduler.block_size)
        hash_block_size = int(scheduler.hash_block_size)
        manager = scheduler.kv_cache_manager
        running = scheduler.running
    except (TypeError, ValueError, AttributeError):
        return False
    hold = globals().get("_glm53_hold_for_inflight_prefix")
    scanned = 0
    for name in ("waiting", "skipped_waiting"):
        queue = getattr(scheduler, name, None)
        if not queue:
            continue
        try:
            iterator = iter(queue)
        except TypeError:
            continue
        for waiter in iterator:
            if scanned >= 4:
                return False
            scanned += 1
            if _glm53_waiter_is_blocked(
                hold, waiter, running, block_size, hash_block_size
            ):
                continue
            hit = _glm53_waiter_cache_hit(waiter, manager)
            try:
                prompt = int(waiter.num_prompt_tokens)
            except (TypeError, ValueError, AttributeError):
                continue
            if _glm53_is_resident_tail(hit, prompt, page, tail):
                return True
    return False


def _glm53_defer_cold_for_resident_tail(scheduler, request) -> bool:
    """Defer one running cold prefill. The counter is stamped only on success."""
    try:
        prompt = int(request.num_prompt_tokens)
        computed = int(request.num_computed_tokens)
        num_running = len(scheduler.running)
        max_running = int(scheduler.max_num_running_reqs)
    except (TypeError, ValueError, AttributeError):
        return False
    left = prompt - computed
    is_tail = _glm53_tail_still_open(request, computed)
    try:
        defer_count = int(getattr(request, "_glm53_resident_tail_defers", 0) or 0)
    except (TypeError, ValueError):
        defer_count = 0
    step_cap = _glm53_resident_tail_steps()
    if is_tail or left <= 64 or defer_count >= step_cap:
        return False
    tail_running = False
    for other in scheduler.running:
        if other is request:
            continue
        try:
            other_computed = int(other.num_computed_tokens)
        except (TypeError, ValueError, AttributeError):
            continue
        if _glm53_tail_still_open(other, other_computed):
            tail_running = True
            break
    tail_waiting = False
    if not tail_running and num_running < max_running:
        tail_waiting = _glm53_has_waiting_resident_tail(
            scheduler,
            _glm53_resident_tail_page(),
            _glm53_resident_tail_tokens(),
        )
    if not _glm53_should_defer_cold(
        prefill_left=left,
        is_tail=is_tail,
        tail_waiting=tail_waiting,
        tail_running=tail_running,
        num_running=num_running,
        max_running=max_running,
        defer_count=defer_count,
        step_cap=step_cap,
    ):
        return False
    try:
        request._glm53_resident_tail_defers = defer_count + 1
    except Exception:
        return False
    try:
        log = globals().get("logger")
        if log is not None:
            log.info(
                "[glm53-resident-tail] defer prefill %s remaining %s",
                getattr(request, "request_id", "?"),
                left,
            )
    except Exception:
        pass
    return True


def _helper_source() -> str:
    names = (
        _glm53_resident_tail_limit,
        _glm53_resident_tail_page,
        _glm53_resident_tail_tokens,
        _glm53_resident_tail_steps,
        _glm53_is_resident_tail,
        _glm53_should_defer_cold,
        _glm53_note_resident_tail,
        _glm53_tail_still_open,
        _glm53_waiter_cache_hit,
        _glm53_waiter_is_blocked,
        _glm53_has_waiting_resident_tail,
        _glm53_defer_cold_for_resident_tail,
    )
    return "\n\n".join(inspect.getsource(fn) for fn in names) + "\n\n\n"


def _write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".glm53-tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _complete(text: str) -> bool:
    return (
        text.count(MARK) == 2
        and text.count("def _glm53_is_resident_tail(") == 1
        and text.count("def _glm53_should_defer_cold(") == 1
        and text.count("def _glm53_note_resident_tail(") == 1
        and text.count("def _glm53_tail_still_open(") == 1
        and text.count("def _glm53_defer_cold_for_resident_tail(") == 1
        and text.count("def _glm53_has_waiting_resident_tail(") == 1
        and text.count(RUN_CALL) == 1
        and text.count(RUN_ANCHOR) == 1
        and text.count(RUN_LPTT_NEW) == 1
        and text.count(RUN_LPTT) == 0
        and text.count(WAIT_LPTT_NEW) == 1
        and text.count(WAIT_LPTT) == 0
        and text.count(CLASS_LINE) == 1
    )


def main() -> int:
    flag = os.environ.get("GLM53_RESIDENT_TAIL", "0").strip()
    if flag == "0":
        print("GLM53_RESIDENT_TAIL is off — installed sources unchanged", flush=True)
        return 0
    if flag != "1":
        print(
            f"GLM53_RESIDENT_TAIL must be exactly 0 or 1 (got: {flag!r})",
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
    if MARK in text or "def _glm53_defer_cold_for_resident_tail(" in text:
        print(f"{MARK} partial install in {SCHEDULER}", file=sys.stderr, flush=True)
        return 1
    counts = {
        "loop": text.count(RUN_ANCHOR),
        "running_cap": text.count(RUN_LPTT),
        "waiting_cap": text.count(WAIT_LPTT),
        "class": text.count(CLASS_LINE),
    }
    if any(n != 1 for n in counts.values()):
        print(
            f"{MARK} anchor drift in {SCHEDULER}: "
            + " ".join(f"{name}={n}" for name, n in counts.items()),
            file=sys.stderr,
            flush=True,
        )
        return 1
    patched = text.replace(CLASS_LINE, _helper_source() + CLASS_LINE, 1)
    patched = patched.replace(RUN_ANCHOR, RUN_ANCHOR + RUN_CALL, 1)
    patched = patched.replace(RUN_LPTT, RUN_LPTT_NEW, 1)
    patched = patched.replace(WAIT_LPTT, WAIT_LPTT_NEW, 1)
    if not _complete(patched):
        print(f"{MARK} patch did not land cleanly in {SCHEDULER}", file=sys.stderr, flush=True)
        return 1
    compile(patched, str(SCHEDULER), "exec")
    _write(SCHEDULER, patched)
    print(f"{MARK} applied to {SCHEDULER}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
