#!/usr/bin/env python3
"""Install the exllamav3 MoE dynamic ticket scheduler (upstream d5e4361) onto the
pinned extension source at image build time.

Upstream exllamav3 commit d5e4361 ("MoE: Replace kernel round-robin assignment
with dynamic ticket scheduler and add dynamic group sizing", 2026-07-06)
replaces the fused `exl3_moe` kernel's static round-robin expert->group
assignment with a self-resetting ticket scheduler: groups claim the next
unclaimed active expert via atomicAdd instead of `idx % concurrency`, so idle
groups steal heavy experts instead of serializing their statically assigned
share. It also makes the group width runtime-configurable (gridDim.x) and
sizes the lock buffer for the scheduler state (MOE_SCHED_INTS).

The kit pins exllamav3 at c5d9c657; this patch cherry-picks d5e4361 onto that
pin (verified: clean apply, +75/-16 over 7 files). Only the 6 ext quant files
are shipped here — the d5e4361 hunk in modules/block_sparse_mlp.py belongs to
exllamav3's own python stack, which the vLLM serve path does not use (the kit
overlay drives exllamav3_ext directly).

Byte-exact three-state installer:
  patched  -> file already matches the vendored post-patch bytes: skip
  pristine -> file matches the pin's bytes: atomic replace with patched
  other    -> anchor drift: FAIL CLOSED, nothing written

The vendored sets live in overlay/exl3-ticket/{pristine,patched}/ next to this
script. The kit's overlay exl3.py already introspects the new trailing
`num_active` parameter (_exl3_moe_accepts_num_active) and passes -1 (unknown)
today, preserving the stock launch geometry; dynamic group widening engages
only when a caller passes a real active count.

Build-time opt-out: GLM53_EXL3_TICKET_SCHEDULER=0 skips the patch entirely
(rollback = previous image tag; the scheduler is compile-time, not a runtime
knob).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

FILES = (
    "exl3_devctx.cu",
    "exl3_devctx.cuh",
    "exl3_moe.cu",
    "exl3_moe.cuh",
    "exl3_moe_common.cuh",
    "exl3_moe_kernel.cuh",
)


def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_exl3_ticket_scheduler.py EXLLAMAV3_EXT")
    ext_root = Path(sys.argv[1]).resolve()
    quant = ext_root / "quant"
    if not quant.is_dir():
        raise SystemExit(f"invalid extension root (no quant/): {ext_root}")

    script_dir = Path(__file__).resolve().parent
    pristine_dir = script_dir / "exl3-ticket" / "pristine"
    patched_dir = script_dir / "exl3-ticket" / "patched"
    for d in (pristine_dir, patched_dir):
        if not d.is_dir():
            raise SystemExit(f"missing vendored set: {d}")

    if os.environ.get("GLM53_EXL3_TICKET_SCHEDULER", "1") == "0":
        print("ticket-scheduler: GLM53_EXL3_TICKET_SCHEDULER=0, skipping")
        return 0

    n_patched = 0
    n_already = 0
    plan: list[tuple[Path, bytes]] = []
    for name in FILES:
        pristine = (pristine_dir / name).read_bytes()
        patched = (patched_dir / name).read_bytes()
        if pristine == patched:
            raise SystemExit(f"vendored sets identical for {name} — packaging bug")
        target = quant / name
        current = target.read_bytes()
        if current == patched:
            n_already += 1
            print(f"ticket-scheduler: {name} already patched")
            continue
        if current != pristine:
            raise SystemExit(
                f"ticket-scheduler FATAL: {target} matches neither the pinned "
                f"pristine bytes ({_sha(pristine)[:12]}) nor the patched bytes "
                f"({_sha(patched)[:12]}); on-disk sha {_sha(current)[:12]} — "
                f"anchor drifted, refusing"
            )
        plan.append((target, patched))

    if plan and n_already:
        # Partial state = the source tree was not pristine to begin with (the
        # installer is run once on a fresh pin tarball at image build).
        # Refuse BEFORE writing anything so the build investigates instead of
        # shipping a surprise.
        raise SystemExit(
            "ticket-scheduler FATAL: mixed pristine/patched state across files "
            "on first application — source tree unexpected, nothing written"
        )
    for target, patched in plan:
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(patched)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        if target.read_bytes() != patched:
            raise SystemExit(f"ticket-scheduler FATAL: post-write verify failed for {target.name}")
        n_patched += 1
        print(f"ticket-scheduler: {target.name} pristine -> patched (atomic)")

    if n_patched and n_already:
        # Partial state = one file drifted into patched-ness while others are
        # pristine. The per-file logic above is safe (every file independently
        # pristine->patched), but a mixed result on a first run means the
        # source tree was not pristine to begin with; refuse loudly so the
        # image build investigates instead of shipping a surprise.
        raise SystemExit(
            "ticket-scheduler FATAL: mixed pristine/patched state across files "
            "on first application — source tree unexpected"
        )
    print(
        f"ticket-scheduler: done (patched={n_patched}, already={n_already}, "
        f"total={len(FILES)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
