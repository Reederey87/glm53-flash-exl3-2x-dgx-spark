#!/usr/bin/env python3
"""Shorten the SHM lost-notify recheck from 5000 ms to 50 ms.

Opt-in via GLM53_SHM_RECHECK_50MS=1. Default off.

Upstream PR #45224 caps SpinCondition.wait at SHM_READER_RECHECK_INTERVAL_MS
so a dropped ZMQ notify (PUB SNDHWM=1, SUB CONFLATE) re-reads the shared-memory
flag instead of parking forever. The constant is not an environment variable.
With busy_loop_s already at 2 ms, a missed second-RPC notify sleeps that whole
cap. 50 ms is one flag check per parked reader. The hot path still returns as
soon as the notify arrives. This patch does not change busy_loop_s.

The replacement keeps the trailing newline so "= 50" is not treated as already
present inside "= 5000". Anchor drift fails closed.
"""
from __future__ import annotations

import os
import sys

PATH = (
    "/usr/local/lib/python3.12/dist-packages/vllm/distributed/"
    "device_communicators/shm_broadcast.py"
)
OLD = "SHM_READER_RECHECK_INTERVAL_MS = 5000\n"
NEW = "SHM_READER_RECHECK_INTERVAL_MS = 50\n"


def rewrite(src: str) -> str:
    """Return src with the recheck constant set to 50 ms.

    Raises ValueError when the assignment anchor is not present exactly once
    and the file is not already rewritten.
    """
    if NEW in src:
        return src
    n = src.count(OLD)
    if n != 1:
        raise ValueError(f"anchor {OLD!r} found {n} times (expected 1)")
    return src.replace(OLD, NEW)


def main() -> int:
    if os.environ.get("GLM53_SHM_RECHECK_50MS", "0") != "1":
        print("shm recheck: GLM53_SHM_RECHECK_50MS!=1, leaving the 5000 ms cap")
        return 0
    with open(PATH, encoding="utf-8") as f:
        src = f.read()
    try:
        out = rewrite(src)
    except ValueError as exc:
        print(f"shm recheck FATAL: {exc}", file=sys.stderr)
        return 1
    if out == src:
        print("shm recheck: already patched (interval=50)")
        return 0
    with open(PATH, "w", encoding="utf-8") as f:
        f.write(out)
    print("shm recheck: SHM_READER_RECHECK_INTERVAL_MS 5000 -> 50 patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
