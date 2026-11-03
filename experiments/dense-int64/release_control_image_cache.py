#!/usr/bin/env python3
"""Release clean file-cache pages only from explicitly identified completed images.

The host must first verify the exact image ID and terminal build, and run this
without GPUs/network with a read-only root. No host model/cache mounts belong
in this container. This does not change files or relax a memory gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from release_build_cache import meminfo, release_tree

IMAGE_ID = "sha256:45ac705ba045e3d35cb7bbb66b807faa82501f29039a215e1e8b1cbd430ea5b8"
CORRECTED_IMAGE_ID = "sha256:40f4c5229d276c1d86c983389bc7281dbbe66b627b6d065a0d2f93bd1c405ab5"
TREATMENT_SHA = "23c70d045e7a2a30fb76d1718eca6d82020c1051d85b5915ee5ef6e84a86a03b"
DENSE_SHA = "313ed8cda2c9665a48e40c88c2676a328ae6174ee65b85514d668d1e9fc1e024"
CANDIDATE_IMAGE_ID = "sha256:0caaeef85b0b267d96aa4f57f07102f818efd42e43b4c0f56d26f8dca77c4b43"
CANDIDATE_TREATMENT_SHA = "0dd9ed8b48147f225f181dd34ca187703843350e5836559583547d1af4e605cc"
CANDIDATE_DENSE_SHA = "89437e814ab6ded8b1e2366149eba4ad9c12a073b67ed153f7b717dfbe96fbd2"
ROOTS = ("/usr", "/opt", "/workspace", "/var/lib/apt/lists")


def verify_container(mountinfo: str, treatment: bytes, dense: bytes, image_id: str = IMAGE_ID) -> None:
    if image_id in (IMAGE_ID, CORRECTED_IMAGE_ID):
        expected_treatment, expected_dense = TREATMENT_SHA, DENSE_SHA
    elif image_id == CANDIDATE_IMAGE_ID:
        expected_treatment, expected_dense = CANDIDATE_TREATMENT_SHA, CANDIDATE_DENSE_SHA
    else:
        raise ValueError("unknown completed image")
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
    if hashlib.sha256(treatment).hexdigest() != expected_treatment:
        raise ValueError("wrong image treatment")
    if hashlib.sha256(dense).hexdigest() != expected_dense:
        raise ValueError("wrong installed dense source")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", choices=(IMAGE_ID, CORRECTED_IMAGE_ID, CANDIDATE_IMAGE_ID), default=IMAGE_ID,
                        help="exact image ID verified by the host before launching this container")
    args = parser.parse_args()
    if sys.platform != "linux":
        raise SystemExit("Linux is required")
    treatment = Path("/workspace/dense-int64.json")
    dense = Path(importlib.metadata.distribution("flashinfer-python").locate_file(
        "flashinfer/experimental/b12x/_lib/dense_gemm.py"))
    if treatment.is_symlink() or dense.is_symlink():
        raise SystemExit("identity files must not be symlinks")
    original = (treatment.read_bytes(), dense.read_bytes())
    verify_container(Path("/proc/self/mountinfo").read_text(), *original, args.image_id)
    before = meminfo()
    reports = {root: release_tree(Path(root)) for root in ROOTS}
    if (treatment.read_bytes(), dense.read_bytes()) != original:
        raise RuntimeError("image identity changed")
    print(json.dumps({"host_verified_image_id": args.image_id,
                      "scopes": reports, "identity_bytes_unchanged": True,
                      "meminfo_before_kib": before, "meminfo_after_kib": meminfo()}, indent=2))
