#!/usr/bin/env python3
"""GB10: shrink SpinCondition busy_loop_s 1 s -> 2 ms (opt-in via GLM53_SPINWAIT_2MS=1).

LOCAL: ported from upstream kit PR #69 (W17 A/B window). The vLLM build in
this image ships shm_broadcast.py with SpinCondition busy_loop_s defaulting
to 1 second: after every read, each reader thread sched_yield()-spins for a
full second before falling back to the zmq notify socket. On GB10 (20-core
Grace, unified memory) those spinning readers compete with NCCL proxy threads
and the engine loop for cores.

Precedent on the same silicon: the DeepSeek-V4-Flash DSpark image was patched
from 1 s to 2 ms in 2026-08 and validated on four Sparks (analysis:
https://nacyot.github.io/artifacts/vllm-spin-wait-gb10/). SpinCondition here
keeps a zmq wake-up path, so shrinking the spin window is safe: readers just
park on the poller 2 ms after the last read instead of 1 s, and the writer's
notify ping wakes them.

Default OFF (no behavior change unless GLM53_SPINWAIT_2MS=1).
Anchor drift fails closed so a vLLM bump cannot silently half-apply.
"""
import os
import sys

PATH = ("/usr/local/lib/python3.12/dist-packages/vllm/distributed/"
        "device_communicators/shm_broadcast.py")
OLD = "busy_loop_s: float = 1,"
NEW = "busy_loop_s: float = 0.002,"

if os.environ.get("GLM53_SPINWAIT_2MS", "0") != "1":
    print("spinwait: GLM53_SPINWAIT_2MS!=1, leaving busy_loop_s as-is")
    sys.exit(0)

with open(PATH, encoding="utf-8") as f:
    src = f.read()

if NEW in src:
    print("spinwait: already patched (busy_loop_s=0.002)")
    sys.exit(0)

n = src.count(OLD)
if n != 1:
    print(f"spinwait FATAL: anchor {OLD!r} found {n} times (expected 1)",
          file=sys.stderr)
    sys.exit(1)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src.replace(OLD, NEW))
print("spinwait: busy_loop_s 1 -> 0.002 patched")
