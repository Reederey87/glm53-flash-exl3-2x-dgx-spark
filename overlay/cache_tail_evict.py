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
"""
from __future__ import annotations

import os

_LOGGED = False
_HOT_LOGGED = False
_REUSED: set[int] = set()


def hot_protect_enabled() -> bool:
    return os.environ.get("GLM53_CACHE_HOT_PROTECT", "0").strip() == "1"


def mark_reused(block_id: int) -> None:
    """Remember that some request has already hit this cached block."""
    _REUSED.add(block_id)


def clear_reused(block_id: int) -> None:
    """Forget a block whose cached tokens are gone."""
    _REUSED.discard(block_id)


def rank_key(
    hashed: bool,
    num_tokens: int | None,
    index: int,
    reused: bool = False,
) -> tuple[int, int, int, int]:
    """Sort key. Smaller is evicted sooner.

    The reused bit is 0 unless the caller passes it. With that bit held at
    0 the order matches depth-only eviction.
    """
    return (1 if hashed else 0, 1 if reused else 0, -(num_tokens or 0), index)


def select_blocks(queue, num_blocks: int) -> list:
    """Remove ``num_blocks`` victims from ``queue`` and return them.

    ``queue`` is the free-block list: ``fake_free_list_head``,
    ``fake_free_list_tail``, and ``remove``. The caller has already checked
    that at least ``num_blocks`` blocks are free.
    """
    global _LOGGED, _HOT_LOGGED
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
    if num_blocks == 0:
        return []
    ranked: list[tuple[tuple[int, int, int, int], object]] = []
    index = 0
    block = queue.fake_free_list_head.next_free_block
    tail = queue.fake_free_list_tail
    while block is not tail:
        reused = bool(
            protect and block.block_hash is not None and block.block_id in _REUSED
        )
        ranked.append(
            (
                rank_key(
                    block.block_hash is not None,
                    block.block_hash_num_tokens,
                    index,
                    reused,
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
