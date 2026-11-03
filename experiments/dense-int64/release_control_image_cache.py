#!/usr/bin/env python3
"""Release clean file-cache pages only from the completed control image.

The host must first verify the exact image ID and terminal build, and run this
without GPUs/network with a read-only root. No host model/cache mounts belong
in this container. This does not change files or relax a memory gate.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from release_build_cache import meminfo, release_tree

IMAGE_ID = "sha256:45ac705ba045e3d35cb7bbb66b807faa82501f29039a215e1e8b1cbd430ea5b8"
TREATMENT_SHA = "23c70d045e7a2a30fb76d1718eca6d82020c1051d85b5915ee5ef6e84a86a03b"
DENSE_SHA = "313ed8cda2c9665a48e40c88c2676a328ae6174ee65b85514d668d1e9fc1e024"
ROOTS = ("/usr", "/opt", "/workspace", "/var/lib/apt/lists")


def verify_container(mountinfo: str, treatment: bytes, dense: bytes) -> None:
    rows = [line.split() for line in mountinfo.splitlines()]
    if any(len(row) < 10 or "-" not in row or len(row) - row.index("-") < 4 for row in rows):
        raise ValueError("malformed mount information")
    roots = [row for row in rows if row[4] == "/"]
    if (len(roots) != 1 or roots[0][3] != "/" or "ro" not in roots[0][5].split(",")
            or roots[0][roots[0].index("-") + 1] != "overlay"):
        raise ValueError("requires one read-only overlay root")
    for row in rows:
        if any(row[4] == root or row[4].startswith(root + "/") for root in ROOTS):
            raise ValueError("external/nested mount inside image scope")
    if hashlib.sha256(treatment).hexdigest() != TREATMENT_SHA:
        raise ValueError("wrong control treatment")
    if hashlib.sha256(dense).hexdigest() != DENSE_SHA:
        raise ValueError("wrong installed dense source")


if __name__ == "__main__":
    if sys.platform != "linux":
        raise SystemExit("Linux is required")
    treatment = Path("/workspace/dense-int64.json")
    dense = Path(importlib.metadata.distribution("flashinfer-python").locate_file(
        "flashinfer/experimental/b12x/_lib/dense_gemm.py"))
    if treatment.is_symlink() or dense.is_symlink():
        raise SystemExit("identity files must not be symlinks")
    original = (treatment.read_bytes(), dense.read_bytes())
    verify_container(Path("/proc/self/mountinfo").read_text(), *original)
    before = meminfo()
    reports = {root: release_tree(Path(root)) for root in ROOTS}
    if (treatment.read_bytes(), dense.read_bytes()) != original:
        raise RuntimeError("control identity changed")
    print(json.dumps({"host_verified_image_id": IMAGE_ID,
                      "scopes": reports, "identity_bytes_unchanged": True,
                      "meminfo_before_kib": before, "meminfo_after_kib": meminfo()}, indent=2))
