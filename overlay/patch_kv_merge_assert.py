#!/usr/bin/env python3
"""Keep KVCacheSpec.merge incompatibility active under ``python -O``.

The day-0 fork still guards the inherited merge with ``assert``. Optimized
Python strips that, so unequal Mamba/GDN specs can be reported as uniform
and collapsed. Upstream #55234 keeps the existing ``AssertionError``
contract used by grouping callers and raises it explicitly.

This overlay ports only that inherited merge. Subclass merges that already
``raise ValueError`` are left alone. The DSpark ``non_causal_multi_token_decode``
any-merge is already present on this image.

Fail closed on drift. Idempotent. Not in the JIT shape hash.
"""
from __future__ import annotations

import ast
import os
import stat
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_KV_CACHE_INTERFACE_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_cache_interface.py",
    )
)
MARK = "# [glm53-kv-merge-assert]"

ANCHOR = """        assert all(spec == specs[0] for spec in specs[1:]), (
            "All layers in the same KV cache group must be the same."
        )
        return copy.deepcopy(specs[0])
"""

PATCHED = """        if not all(spec == specs[0] for spec in specs[1:]):
            raise AssertionError(  # [glm53-kv-merge-assert]
                "All layers in the same KV cache group must be the same."
            )
        return copy.deepcopy(specs[0])
"""


def verified_state(source: str) -> bool:
    return (
        ANCHOR not in source
        and PATCHED in source
        and source.count(MARK) == 1
        and source.count("raise AssertionError(") >= 1
    )


def inherited_merge_state(source: str) -> str:
    """Classify inherited merge the same way the installer will apply it."""
    if verified_state(source):
        return "raise"
    try:
        _, action = prepare(source)
    except ValueError:
        return "unknown"
    if action == "patched" and source.count(ANCHOR) == 1:
        return "assert"
    return "unknown"


def prepare(source: str) -> tuple[str, str]:
    if MARK in source:
        if not verified_state(source):
            raise ValueError(
                "glm53-kv-merge-assert marker present but the patched merge "
                "is incomplete; refusing to guess"
            )
        return source, "already present"
    if verified_state(source):
        return source, "already patched"
    if source.count(ANCHOR) != 1:
        raise ValueError(
            "inherited KVCacheSpec.merge assert drifted "
            f"(anchor={source.count(ANCHOR)})"
        )
    patched = source.replace(ANCHOR, PATCHED, 1)
    if not verified_state(patched):
        raise ValueError("kv-merge-assert post-patch verification failed")
    return patched, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-kv-merge-assert.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob("kv_cache_interface*.pyc"):
        pyc.unlink(missing_ok=True)


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    source = TARGET.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"kv-merge-assert preflight failed: {exc}") from exc
    ast.parse(patched, filename=str(TARGET))
    compile(patched, str(TARGET), "exec")
    if patched != source:
        replace_file(TARGET, patched)
        clear_pyc(TARGET)
    print(f"{TARGET.name}: kv-merge-assert {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
