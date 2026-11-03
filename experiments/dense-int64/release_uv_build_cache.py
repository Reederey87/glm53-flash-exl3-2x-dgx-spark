#!/usr/bin/env python3
"""Advise clean pages from one identified, idle BuildKit uv cache mount.

Host guards must confirm no builds or serving processes before invocation.
Run CPU-only with read-only mounts; never run uv itself against this cache.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from release_build_cache import meminfo, release_tree

CACHE_ID = "5mfrweath3j5gje6zh1y8z4td"
SOURCE = f"/var/lib/docker/overlay2/{CACHE_ID}/diff"
ROOT = Path("/completed-uv")


def verify_cache(record: dict, mountinfo: str) -> None:
    description = record.get("Description", "")
    if (record.get("ID") != CACHE_ID or record.get("Type") != "exec.cachemount"
            or record.get("Reclaimable") is not True or record.get("Shared") is not False
            or not description.startswith("cached mount /root/.cache/uv from exec ")
            or not description.endswith('with id "/uv-cache"')):
        raise ValueError("not the identified inactive uv cache")
    rows = [line.split() for line in mountinfo.splitlines()]
    matches = [row for row in rows if len(row) >= 10 and row[4] == str(ROOT)]
    if (len(matches) != 1 or matches[0][3] != SOURCE
            or "ro" not in matches[0][5].split(",")
            or any(len(row) >= 5 and row[4].startswith(str(ROOT) + "/") for row in rows)):
        raise ValueError("requires the exact cache mounted read-only without nested mounts")


if __name__ == "__main__":
    if sys.platform != "linux":
        raise SystemExit("Linux is required")
    record = json.loads(Path("/checks/cache-record.json").read_text())
    verify_cache(record, Path("/proc/self/mountinfo").read_text())
    tag = ROOT / "CACHEDIR.TAG"
    if tag.is_symlink() or not tag.read_bytes().startswith(b"Signature: 8a477f597d28d172789f06886806bc55d"):
        raise SystemExit("missing cache signature")
    for name in ("archive-v0", "wheels-v7"):
        path = ROOT / name
        if path.is_symlink() or not path.is_dir():
            raise SystemExit("unexpected uv cache layout")
    before = meminfo()
    report = release_tree(ROOT)
    report.update(cache_id=CACHE_ID, meminfo_before_kib=before, meminfo_after_kib=meminfo())
    print(json.dumps(report, indent=2))
