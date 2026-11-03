#!/usr/bin/env python3
"""Preserve this experiment's compiler caches; call only after both-node idle gate.

No file contents are deleted. Checkpoint caches are outside the fixed scope.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import stat
import sys

CACHES = (".cache/vllm", ".cache/flashinfer", ".cache/b12x", ".triton", ".tilelang")
ARMS = ("incumbent", "control", "candidate")


def preserve(root: Path, arm: str) -> list[dict]:
    if arm not in ARMS:
        raise ValueError("unknown serving arm")
    if not root.is_dir() or root.resolve() != root:
        raise ValueError("requires a direct absolute home directory")
    planned = []
    for relative in CACHES:
        source = root / relative
        target = source.with_name(source.name + f".int64-before-{arm}-20261103")
        if source.resolve() != source or not source.is_dir():
            raise ValueError(f"cache is missing, linked or not a directory: {source}")
        before = source.stat()
        if before.st_uid != os.geteuid() or before.st_gid != os.getegid():
            raise ValueError(f"cache ownership differs from runtime user: {source}")
        if target.exists() or target.is_symlink():
            raise ValueError(f"backup already exists: {target}")
        planned.append((source, target, before))

    moved = []
    try:
        for source, target, before in planned:
            source.rename(target)
            moved.append((source, target))
            source.mkdir(mode=stat.S_IMODE(before.st_mode))
            source.chmod(stat.S_IMODE(before.st_mode))
            after = target.stat()
            if (before.st_dev, before.st_ino, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_mtime_ns
            ):
                raise RuntimeError(f"preserved directory identity changed: {target}")
            if any(source.iterdir()):
                raise RuntimeError(f"new compiler cache is not empty: {source}")
    except Exception:
        # Remove only newly created EMPTY directories, then restore originals.
        # rmdir refuses a concurrent writer's data instead of deleting it.
        for source, target in reversed(moved):
            if source.exists():
                source.rmdir()
            target.rename(source)
        raise
    return [
        {"cache": str(source), "preserved": str(target), "device": before.st_dev,
         "inode": before.st_ino, "mode": oct(stat.S_IMODE(before.st_mode)),
         "uid": before.st_uid, "gid": before.st_gid, "fresh_empty": True}
        for source, target, before in planned
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or os.uname().machine != "aarch64":
        raise SystemExit("requires the ARM64 Linux runtime node")
    if pwd.getpwuid(os.geteuid()).pw_name != "nvidia":
        raise SystemExit("requires runtime owner nvidia")
    with args.receipt.open("x") as stream:
        try:
            result = {"ok": True, "arm": args.arm,
                      "caches": preserve(Path("/home/nvidia"), args.arm)}
        except Exception as exc:
            json.dump({"ok": False, "error": str(exc)}, stream, indent=2)
            stream.write("\n")
            raise
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
