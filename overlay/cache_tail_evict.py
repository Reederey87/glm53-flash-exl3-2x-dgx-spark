"""Pick prefix-cache victims by depth, then by whether the page was hit.

vLLM already frees one request's blocks tail-first: the block that covers
more tokens is less often a shared prefix, so it is pushed closer to the
eviction end. Across requests the queue is still least-recently freed, so
the first page of an agent that is paused on a tool call is taken before a
deeper page of a request that finished later.

``select_blocks`` keeps that unhashed-first rule and, among cached blocks,
takes the greatest ``block_hash_num_tokens`` first. A missing tail shortens
the hit. A missing first page drops it. Equal depth keeps queue order.

``GLM53_CACHE_HOT_PROTECT=1`` adds one more key. A block recorded by
``mark_reused`` (a later request hit that hash) is evicted only after every
one-shot block. Depth still orders each of those two bands, so the first
return of a paused agent keeps the tail-first behavior. ``clear_reused``
runs when the hash is dropped, because the physical block is about to hold
different tokens.

``GLM53_TOOL_RETURN_GRACE=1`` adds a band between those two. The engine
snapshots a request's hashed block ids when it frees them. The chat layer
confirms the snapshot only after it rewrites ``finish_reason`` to
``tool_calls``, which is after the free and before the client runs the
tool. Until that confirm, and for a plain stop, the pages stay one-shot.
Grace is a lazy 600 second TTL checked at eviction, capped at a quarter of
the 566 usable blocks. The lease has to outlast a competing prefill on
this pool, not only the tool call itself: the eighteen-prompt flood takes
about seven minutes, and a 120 second lease expired while that flood was
still running. The cap drops the oldest mark first and, within one
timestamp, the deepest page, so a prefix head outlives its own tail. A
later real hit promotes the block through hot-protect and clears grace.
No timer thread and no hard pin.
"""
from __future__ import annotations

import os
import time

_LOGGED = False
_HOT_LOGGED = False
_GRACE_LOGGED = False
_REUSED: set[int] = set()
# block_id -> (monotonic timestamp, hash token count)
_GRACE: dict[int, tuple[float, int]] = {}
# internal request id -> (external request id, block/hash pairs, free time).
# One bounded mapping owns all pending state; no separate recent-free queue.
_SNAPSHOTS: dict[str, tuple[str, list[tuple[int, object, int]], float]] = {}
GRACE_TTL_S = 600.0
GRACE_CAP = 141  # quarter of the 566 usable blocks
_SNAPSHOT_LIMIT = 16


def hot_protect_enabled() -> bool:
    return os.environ.get("GLM53_CACHE_HOT_PROTECT", "0").strip() == "1"


def tool_grace_enabled() -> bool:
    return os.environ.get("GLM53_TOOL_RETURN_GRACE", "0").strip() == "1"


def mark_reused(block_id: int) -> None:
    """Remember that some request has already hit this cached block."""
    _REUSED.add(block_id)
    _GRACE.pop(block_id, None)


def clear_reused(block_id: int) -> None:
    """Forget a block whose cached tokens are gone."""
    _REUSED.discard(block_id)
    _GRACE.pop(block_id, None)


def reset_grace_state() -> None:
    """Drop grace marks and snapshots. Tests use this between cases."""
    _GRACE.clear()
    _SNAPSHOTS.clear()


def remember_snapshot(
    request_id: str, pairs: list[tuple[int, object, int]],
    external_request_id: str | None = None,
) -> None:
    """Keep hashed ids from a request that is about to be freed.

    The confirm arrives later, from the process that rewrites the finish
    reason. Flag off stores nothing. EngineCoreRequest.external_req_id is
    carried into the scheduler Request explicitly: timing never establishes
    request identity. Multiple pending requests with the same external id
    are ambiguous and cannot be confirmed.
    """
    if not tool_grace_enabled():
        return
    now = time.monotonic()
    _expire_snapshots(now)
    _SNAPSHOTS.pop(request_id, None)
    if not pairs:
        return
    stored = list(pairs)
    _SNAPSHOTS[request_id] = (external_request_id or request_id, stored, now)
    while len(_SNAPSHOTS) > _SNAPSHOT_LIMIT:
        _SNAPSHOTS.pop(next(iter(_SNAPSHOTS)))
    print(
        f"[glm53-tool-return-grace] stored {len(stored)} pages",
        flush=True,
    )


def note_grace(block_id: int, num_tokens: int, now: float | None = None) -> None:
    """Put one block in the grace band and enforce the cap."""
    _GRACE[block_id] = (time.monotonic() if now is None else now, num_tokens)
    _trim_grace()


def confirm_tool_grace(request_id: str, pool) -> int:
    """Mark snapshotted blocks whose hash is still the one we freed.

    A block that was overwritten since the free is left one-shot. Returns
    how many blocks were marked.
    """
    if not tool_grace_enabled():
        return 0
    now = time.monotonic()
    _expire_snapshots(now)
    matches = [key for key, (external, _, _) in _SNAPSHOTS.items()
               if external == request_id]
    if len(matches) != 1:
        # Fail closed, including duplicate external ids. Never let a second
        # confirm claim the remaining sibling after an ambiguous first one.
        for key in matches:
            _SNAPSHOTS.pop(key)
        print(
            f"[glm53-tool-return-grace] no unique snapshot (matches={len(matches)})",
            flush=True,
        )
        return 0
    _, pairs, _ = _SNAPSHOTS.pop(matches[0])
    blocks = pool.blocks
    marked = 0
    for block_id, hashed, num_tokens in pairs:
        if not isinstance(block_id, int) or block_id < 0 or block_id >= len(blocks):
            continue
        block = blocks[block_id]
        if block.block_hash != hashed:
            continue
        _GRACE[block_id] = (now, num_tokens or 0)
        marked += 1
    _trim_grace()
    print(
        f"[glm53-tool-return-grace] confirmed {marked} of {len(pairs)} pages",
        flush=True,
    )
    return marked


def _expire_snapshots(now: float) -> None:
    for key, (_, _, stamp) in list(_SNAPSHOTS.items()):
        if now - stamp > GRACE_TTL_S:
            _SNAPSHOTS.pop(key)


def grace_live(block_id: int, now: float) -> bool:
    """True when the grace mark exists and is inside the TTL.

    An expired mark is cleared here, so the block falls back to one-shot
    without a timer thread.
    """
    row = _GRACE.get(block_id)
    if row is None:
        return False
    if now - row[0] > GRACE_TTL_S:
        _GRACE.pop(block_id, None)
        return False
    return True


def _trim_grace() -> None:
    overflow = len(_GRACE) - GRACE_CAP
    if overflow <= 0:
        return
    # Oldest mark first. The same timestamp drops the deepest page first,
    # so the head of a prefix is what the cap keeps.
    victims = sorted(_GRACE.items(), key=lambda item: (item[1][0], -item[1][1]))
    for block_id, _row in victims[:overflow]:
        _GRACE.pop(block_id, None)


def rank_key(
    hashed: bool,
    num_tokens: int | None,
    index: int,
    reused: bool = False,
    grace: bool = False,
) -> tuple[int, int, int, int]:
    """Sort key. Smaller is evicted sooner.

    Band 0 is one-shot, band 1 is an unexpired tool-return, band 2 is a
    block a later request has hit. With both bits held at false the order
    matches depth-only eviction. A reused block outranks grace.
    """
    if reused:
        band = 2
    elif grace:
        band = 1
    else:
        band = 0
    return (1 if hashed else 0, band, -(num_tokens or 0), index)


def select_blocks(queue, num_blocks: int) -> list:
    """Remove ``num_blocks`` victims from ``queue`` and return them.

    ``queue`` is the free-block list: ``fake_free_list_head``,
    ``fake_free_list_tail``, and ``remove``. The caller has already checked
    that at least ``num_blocks`` blocks are free.
    """
    global _LOGGED, _HOT_LOGGED, _GRACE_LOGGED
    if not _LOGGED:
        _LOGGED = True
        print(
            "[glm53-cache-tail-evict] deepest cached block is evicted first",
            flush=True,
        )
    protect = hot_protect_enabled()
    if protect and not _HOT_LOGGED:
        _HOT_LOGGED = True
        print(
            "[glm53-cache-hot-protect] a prefix that has been hit "
            "is evicted after one-shot blocks",
            flush=True,
        )
    grace_on = tool_grace_enabled()
    if grace_on and not _GRACE_LOGGED:
        _GRACE_LOGGED = True
        print(
            "[glm53-tool-return-grace] a tool-call page is evicted after "
            "one-shot blocks and before a reused page",
            flush=True,
        )
    if num_blocks == 0:
        return []
    now = time.monotonic()
    ranked: list[tuple[tuple[int, int, int, int], object]] = []
    index = 0
    block = queue.fake_free_list_head.next_free_block
    tail = queue.fake_free_list_tail
    while block is not tail:
        reused = bool(
            protect and block.block_hash is not None and block.block_id in _REUSED
        )
        grace = bool(
            grace_on
            and not reused
            and block.block_hash is not None
            and grace_live(block.block_id, now)
        )
        ranked.append(
            (
                rank_key(
                    block.block_hash is not None,
                    block.block_hash_num_tokens,
                    index,
                    reused,
                    grace,
                ),
                block,
            )
        )
        block = block.next_free_block
        index += 1
    ranked.sort(key=lambda item: item[0])
    chosen = [item[1] for item in ranked[:num_blocks]]
    for block in chosen:
        queue.remove(block)
    return chosen
