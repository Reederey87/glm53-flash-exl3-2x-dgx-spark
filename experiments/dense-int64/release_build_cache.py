#!/usr/bin/env python3
"""Advise away clean cache pages from a completed, read-only build-layer mount.

No file data is read or written. This is not a global cache purge and does not
change the memory gate. The host controller must first verify the build ended.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


def completed_mount(workspace: str, mountinfo: str) -> Path:
    allowed = {"vllm": "vllm/vllm", "flashinfer": "flashinfer"}
    if workspace not in allowed:
        raise ValueError("unknown completed workspace")
    root = Path(f"/completed-{workspace}")
    rows = [line.split() for line in mountinfo.splitlines()]
    matches = [row for row in rows if len(row) >= 10 and row[4] == str(root)]
    source = rf"/var/lib/docker/overlay2/[A-Za-z0-9_-]+/diff/workspace/{allowed[workspace]}"
    if (len(matches) != 1 or "ro" not in matches[0][5].split(",")
            or re.fullmatch(source, matches[0][3]) is None):
        raise ValueError("requires the completed Docker build workspace mounted read-only")
    return root


def release_tree(root: Path) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("expected a real directory")
    digest = hashlib.sha256()
    count = size = skipped = 0
    seen = set()
    for directory, subdirs, files in os.walk(root, followlinks=False):
        subdirs[:] = sorted(name for name in subdirs
                            if not (Path(directory) / name).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if not stat.S_ISREG(path.lstat().st_mode):
                skipped += 1
                continue
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError("file changed type")
                identity = (before.st_dev, before.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                # size=0 advises the entire file; dirty pages may remain.
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                after = os.fstat(fd)
                fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
                if any(getattr(before, key) != getattr(after, key) for key in fields):
                    raise RuntimeError("build artifact metadata changed during advice")
                record = [str(path.relative_to(root)), before.st_size, before.st_mtime_ns]
                digest.update((json.dumps(record, separators=(",", ":")) + "\n").encode())
                count += 1
                size += before.st_size
            finally:
                os.close(fd)
    return {"files_advised": count, "file_bytes_advised": size,
            "non_regular_files_skipped": skipped,
            "metadata_sha256": digest.hexdigest(), "metadata_unchanged": True,
            "note": "Advisory release of clean pages; byte count is file size, not measured reclaimed memory."}


def meminfo() -> dict:
    wanted = {"MemFree", "MemAvailable", "Cached", "Dirty", "Writeback"}
    return {line.split(":", 1)[0]: int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.split(":", 1)[0] in wanted}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", choices=("vllm", "flashinfer"), default="vllm")
    args = parser.parse_args()
    if sys.platform != "linux" or not hasattr(os, "posix_fadvise"):
        raise SystemExit("Linux posix_fadvise is required")
    root = completed_mount(args.workspace, Path("/proc/self/mountinfo").read_text())
    before = meminfo()
    report = release_tree(root)
    report.update(workspace=args.workspace, meminfo_before_kib=before, meminfo_after_kib=meminfo())
    print(json.dumps(report, indent=2))
