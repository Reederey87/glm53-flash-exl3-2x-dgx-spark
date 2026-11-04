#!/usr/bin/env python3
"""Advise clean pages from the exact completed vLLM wheel export, preserving it."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from release_build_cache import meminfo

SOURCE = "/home/nvidia/glm53-source-builds/dense-int64-explicit-fetch/context/artifacts/vllm-wheels"
SOURCES = (
    SOURCE,
    "/home/nvidia/glm53-source-builds/dense-int64-2157b70/context/artifacts/vllm-wheels",
    "/home/nvidia/glm53-source-builds/dense-int64-2157b70-resume1/context/artifacts/vllm-wheels",
)
ROOT = Path("/completed-export")
WHEEL = "vllm-0.1.dev21830+g8244d1c44.d20261103-cp312-cp312-linux_aarch64.whl"
SHA = "1ee0eb5863106d0dbfa70b1a9b5f7bfc05776d8927a4e694f8af23557670ade0"
SIZE = 502172556


def verify_mount(mountinfo: str) -> str:
    rows = [line.split() for line in mountinfo.splitlines()]
    matches = [row for row in rows if len(row) >= 10 and row[4] == str(ROOT)]
    if (len(matches) != 1 or matches[0][3] not in SOURCES
            or "ro" not in matches[0][5].split(",")
            or any(len(row) >= 5 and row[4].startswith(str(ROOT) + "/") for row in rows)):
        raise ValueError("requires the exact completed export mounted read-only without nested mounts")
    return matches[0][3]


def release(path: Path) -> dict:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != SIZE:
            raise ValueError("unexpected exported wheel type or size")
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != SHA:
            raise ValueError("exported wheel hash differs from the verified build")
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        after = os.fstat(fd)
        if any(getattr(before, key) != getattr(after, key)
               for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")):
            raise RuntimeError("export metadata changed during advice")
        return {"sha256": SHA, "file_bytes_advised": before.st_size,
                "metadata_unchanged": True,
                "note": "Advisory only; file size is not measured reclaimed memory."}
    finally:
        os.close(fd)


if __name__ == "__main__":
    if sys.platform != "linux":
        raise SystemExit("Linux is required")
    source = verify_mount(Path("/proc/self/mountinfo").read_text())
    before = meminfo()
    report = release(ROOT / WHEEL)
    report.update(source=source, meminfo_before_kib=before, meminfo_after_kib=meminfo())
    print(json.dumps(report, indent=2))
