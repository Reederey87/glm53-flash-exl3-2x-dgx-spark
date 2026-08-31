#!/usr/bin/env python3
"""[glm53-fgapc] Fine-grained APC for the GLM-5.3 hybrid: exempt the KpoolTail
scratch manager from the partial-hash-hit veto.

LOCAL: ported from upstream kit PR #59 (W18 A/B window). Upstream disables
fine-grained prefix-cache hits whenever ANY cache manager lacks
`supports_fine_grained_hash_lookup` and is not already hash-grained. On this
hybrid the only such manager is KpoolTailManager — a one-block circular
scratch per request that (a) never has block hashes computed, (b) already
opts out of the hybrid hit min (its find_longest_cache_hit returns 0), and
(c) whose group is reseeded fresh when the cached window does not cover the
hit. Its veto therefore zeroes a capability the MLA + mamba(align) managers
do support (hash grain = gcd of participating groups = 64 here), and every
follow-up re-prefills up to a full 3584-token hybrid block. Exempting it
lets the MLA/mamba hit reconcile at hash granularity.

Gated by GLM53_FINE_GRAINED_APC (start.sh default 0 here — flipped to 1 for
the window; env-only, not in the JIT shape hash). Idempotent via the
[glm53-fgapc] marker; ast.parse-validated after edit; fails closed on
anchor drift.
"""

import ast
import os
import sys

MARKER = "[glm53-fgapc]"

VETO_OLD = """            unsupported_partial_hit_managers = {
                type(manager).__name__
                for manager in self.single_type_managers
                if not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size
            }"""

VETO_NEW = """            unsupported_partial_hit_managers = {
                type(manager).__name__
                for manager in self.single_type_managers
                if not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size
                # [glm53-fgapc] KpoolTail is a 1-block/req scratch: it never
                # sees block hashes and already opts out of the hybrid hit
                # min, so its block-aligned-only lookup must not veto
                # fine-grained hits for the managers that do participate.
                and type(manager).__name__ != "KpoolTailManager"
            }"""

DIAG_OLD = """        self.verify_and_split_kv_cache_groups()"""

DIAG_NEW = """        logger.info(
            # [glm53-fgapc]
            "[glm53-fgapc] partial_hash=%s hash_block=%s managers=%s",
            self.enable_partial_hash_hits,
            hash_block_size,
            sorted(
                (
                    type(manager).__name__,
                    manager.block_size,
                    bool(
                        getattr(
                            manager, "supports_fine_grained_hash_lookup", False
                        )
                    ),
                )
                for manager in self.single_type_managers
            ),
        )
        self.verify_and_split_kv_cache_groups()"""


def patch_file(path: str, dry_run: bool = False) -> int:
    with open(path, encoding="utf-8") as f:
        text = f.read()

    if MARKER in text:
        # Idempotency must mean "both edits fully present", not "marker seen".
        if VETO_NEW in text and DIAG_NEW in text:
            ast.parse(text, filename=path)
            print(f"[patch_fine_grained_apc] {path}: already patched; no-op.")
            return 0
        raise AssertionError(
            f"{path}: marker present but the patched forms are incomplete "
            "(partial/interrupted earlier edit?); refusing to touch it."
        )

    if VETO_OLD not in text:
        raise AssertionError(
            f"{path}: partial-hash veto block not found (upstream layout "
            "changed?); refusing to guess."
        )
    text = text.replace(VETO_OLD, VETO_NEW, 1)

    if text.count(DIAG_OLD) != 1:
        raise AssertionError(
            f"{path}: expected exactly one verify_and_split call site, "
            f"found {text.count(DIAG_OLD)}."
        )
    text = text.replace(DIAG_OLD, DIAG_NEW, 1)

    try:
        ast.parse(text, filename=path)
    except SyntaxError as e:
        raise AssertionError(f"POST-EDIT ast.parse FAILED for {path}: {e}") from e

    if dry_run:
        print(f"[patch_fine_grained_apc] DRY RUN -- {path} not written.")
    else:
        tmp = path + ".glm53-fgapc.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)  # atomic: never leave a marker-bearing partial file
        print(f"[patch_fine_grained_apc] {path}: fine-grained APC exemption applied.")
    return 1


def main() -> int:
    # LOCAL: default OFF until the W18 window adopts it (upstream PR #59 ships
    # it on). GLM53_FINE_GRAINED_APC=1 applies; anything else leaves the
    # block-aligned-only upstream behaviour.
    if os.environ.get("GLM53_FINE_GRAINED_APC", "0") != "1":
        print(
            "[patch_fine_grained_apc] GLM53_FINE_GRAINED_APC!=1 — "
            "skipping (block-aligned-only upstream behavior)."
        )
        return 0

    # GLM53_FGAPC_TARGET is a test hook only (synthetic source files).
    path = os.environ.get(
        "GLM53_FGAPC_TARGET",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py",
    )
    if not os.path.isfile(path):
        # Enabled but nothing to patch: fail closed — start.sh's set -e stops the
        # rank before `exec vllm serve` instead of booting silently unpatched.
        print(f"[patch_fine_grained_apc] FATAL: target not found: {path}", file=sys.stderr)
        return 1
    patch_file(path, dry_run=os.environ.get("DRY_RUN") == "1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
