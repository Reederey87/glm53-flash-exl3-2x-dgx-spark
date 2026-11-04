#!/usr/bin/env python3
"""Apply the frozen kernel treatment inside the full FlashInfer source build."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply(source: Path, bundle: Path, output: Path, arm: str) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    patch = bundle / "kernel.patch"
    if digest(patch) != manifest["patch_sha256"]:
        raise ValueError("kernel patch differs from frozen manifest")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != manifest["flashinfer_base"]:
        raise ValueError(f"unexpected FlashInfer source: {actual}")
    for item in manifest["files"]:
        if digest(source / item["path"]) != item["base_sha256"]:
            raise ValueError(f"base file differs: {item['path']}")
    if arm == "candidate":
        subprocess.run(["git", "apply", "--check", str(patch.resolve())], cwd=source, check=True)
        subprocess.run(["git", "apply", str(patch.resolve())], cwd=source, check=True)
    key = "candidate_sha256" if arm == "candidate" else "base_sha256"
    for item in manifest["files"]:
        if digest(source / item["path"]) != item[key]:
            raise ValueError(f"result differs: {item['path']}")
    receipt = {"arm": arm, "flashinfer_commit": actual,
               "files": {item["path"]: item[key] for item in manifest["files"]},
               "patch_sha256": manifest["patch_sha256"], "gpu_qualified": False}
    output.mkdir(parents=True, exist_ok=True)
    (output / ".dense-int64.json").write_text(json.dumps(receipt, indent=2) + "\n")
    # Tests are source artifacts, not a claim of native compilation. They run
    # against the wheel installed in the runner, not an overlaid Python tree.
    shutil.copytree(source / "tests", output / "kernel-tests" / "tests")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("control", "candidate"))
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("/workspace/wheels"))
    args = parser.parse_args()
    print(json.dumps(apply(args.source, args.bundle, args.output, args.arm), indent=2))
