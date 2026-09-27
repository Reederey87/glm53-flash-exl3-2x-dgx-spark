"""Pick prefix-cache victims by depth, not only by free order.

vLLM already frees one request's blocks tail-first: the block that covers
more tokens is less often a shared prefix, so it is pushed closer to the
eviction end. Across requests the queue is still least-recently freed, so
the first page of an agent that is paused on a tool call is taken before a
deeper page of a request that finished later.

``select_blocks`` keeps that unhashed-first rule and, among cached blocks,
takes the greatest ``block_hash_num_tokens`` first. A missing tail shortens
the hit. A missing first page drops it. Equal depth keeps queue order.
"""
from __future__ import annotations

_LOGGED = False


def rank_key(hashed: bool, num_tokens: int | None, index: int) -> tuple[int, int, int]:
    """Sort key. Smaller is evicted sooner."""
    return (1 if hashed else 0, -(num_tokens or 0), index)


def select_blocks(queue, num_blocks: int) -> list:
    """Remove ``num_blocks`` victims from ``queue`` and return them.

    ``queue`` is the free-block list: ``fake_free_list_head``,
    ``fake_free_list_tail``, and ``remove``. The caller has already checked
    that at least ``num_blocks`` blocks are free.
    """
    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        print(
            "[glm53-cache-tail-evict] deepest cached block is evicted first",
            flush=True,
        )
    if num_blocks == 0:
        return []
    ranked: list[tuple[tuple[int, int, int], object]] = []
    index = 0
    block = queue.fake_free_list_head.next_free_block
    tail = queue.fake_free_list_tail
    while block is not tail:
        ranked.append(
            (
                rank_key(
                    block.block_hash is not None,
                    block.block_hash_num_tokens,
                    index,
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
