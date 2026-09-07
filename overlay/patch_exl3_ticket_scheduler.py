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

The kit currently pins exllamav3 at v1.4.7 (`ca13bdd`), which already contains
the ticket scheduler. This installer is retained for the historical c5d9c657
rollback pin: it cherry-picks d5e4361 onto that pin (verified: clean apply,
+75/-16 over 7 files). Only the 6 ext quant files are shipped here — the
d5e4361 hunk in modules/block_sparse_mlp.py belongs to exllamav3's own python
stack, which the vLLM serve path does not use (the kit overlay drives
exllamav3_ext directly).

Byte-exact four-state installer:
  native   -> every quant file SHA256-matches the pinned v1.4.7 (ca13bdd)
              set, including exl3_moe.cuh which is byte-identical to the
              historical patched header: skip, write nothing
  patched  -> file already matches the vendored post-patch bytes: skip
  pristine -> file matches the c5d9c657 pin's bytes: atomic replace with patched
  other    -> mixed native/c5d9 or ordinary drift: FAIL CLOSED, nothing written

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

# Semantic markers that prove a kernel has d5e4361-class ticket scheduling.
# Host tests assert these on the vendored patched set; native skip is hash-exact.
NATIVE_TICKET_MARKERS = (
    "atomicAdd(&sched[0], 1)",
    "expert_idx_assign++ != ticket",
    "MOE_SCHED_OFFSET",
    "num_active",
)

# Exact v1.4.7 (ca13bdd) ext/quant SHA256. exl3_moe.cuh is shared with the
# historical patched header (96f4fc24…); the other five files are not.
NATIVE_V147_SHA256 = {
    "exl3_devctx.cu": "545e1909873b2bd8f6cf598edce0c7772519e2d6e4b473fc72019e678379a34a",
    "exl3_devctx.cuh": "effb1827e9b6c61ba95287ccfa5d6a0b6ccef103c0d588d992359be9291834c0",
    "exl3_moe.cu": "ce394daf0cebafd65e848dffb374562554560b38f52b7070df5d782b1488b218",
    "exl3_moe.cuh": "96f4fc2473474986cd4494554355efe2079a3b717c8612eadf5193334cfbd15f",
    "exl3_moe_common.cuh": "b3c4e2399b8cf5a9c038dbe06da0aa8e4a3500dcd838e1c4a0ac95b22ce2acd6",
    "exl3_moe_kernel.cuh": "6f58cfa0f66557c867ad821ffe9f165da0b3e760eb8633f734b8dbd95696501a",
}
NATIVE_V147_SHARED_HEADER = "exl3_moe.cuh"


def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def tree_is_native_v147(quant: Path) -> bool:
    """True when every FILE SHA256-matches the pinned v1.4.7 quant set."""
    for name in FILES:
        path = quant / name
        if not path.is_file():
            return False
        if _sha(path.read_bytes()) != NATIVE_V147_SHA256[name]:
            return False
    return True


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

    if tree_is_native_v147(quant):
        print(
            "ticket-scheduler: native v1.4.7 (ca13bdd) quant set present, "
            "including shared historical exl3_moe.cuh — skipping byte-exact installer"
        )
        print("ticket-scheduler: done (native=1, patched=0, already=0, total=6)")
        return 0

    n_patched = 0
    n_already = 0
    plan: list[tuple[Path, bytes]] = []
    n_pristine = 0
    native_exclusive: list[str] = []
    drifted: list[Path] = []
    patched_exclusive: list[str] = []
    for name in FILES:
        pristine = (pristine_dir / name).read_bytes()
        patched = (patched_dir / name).read_bytes()
        if pristine == patched:
            raise SystemExit(f"vendored sets identical for {name} — packaging bug")
        target = quant / name
        current = target.read_bytes()
        current_sha = _sha(current)
        if current_sha == NATIVE_V147_SHA256[name] and name != NATIVE_V147_SHARED_HEADER:
            native_exclusive.append(name)
        if current == patched:
            n_already += 1
            if name != NATIVE_V147_SHARED_HEADER:
                patched_exclusive.append(name)
            print(f"ticket-scheduler: {name} already patched")
            continue
        if current == pristine:
            n_pristine += 1
            plan.append((target, patched))
            continue
        drifted.append(target)

    if native_exclusive and (n_pristine or patched_exclusive or drifted):
        raise SystemExit(
            "ticket-scheduler FATAL: mixed native/c5d9c657 state across files "
            "— source tree unexpected, nothing written"
        )

    if drifted:
        raise SystemExit(
            f"ticket-scheduler FATAL: {drifted[0]} matches neither the pinned "
            "pristine bytes nor the patched bytes and is not the native v1.4.7 "
            "quant set — anchor drifted, refusing"
        )

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
