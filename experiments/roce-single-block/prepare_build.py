#!/usr/bin/env python3
"""Create an isolated full-source Docker context; does not build or deploy."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

UPSTREAM_COMMIT = "0bdc9c94cd89843e1c8f7d7b33df2d5bc746c03d"
DOCKERFILE_SHA = "71d5a832a97f3e2b540cec1bf03d0c843cbeb984d8309adcdf580dc703630c1e"
CUDA_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:450d11555d20ac8ebbbc13ebf17589c2bd42869171a90179ce7098b4a5e64c6a"
NCCL_COMMIT = "12df1a11afad322be5a204a2db890161cbf8131d"
VLLM_COMMIT = "8244d1c44348ae8a5bbfc9f1b607021652283ed4"
FLASHINFER_COMMIT = "91271757f7746f3dfb58f7c2aeb44a663389de9e"


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one Dockerfile anchor: {old!r}")
    return text.replace(old, new, 1)


def prepare(repo: Path, destination: Path) -> dict:
    if destination.exists():
        raise ValueError("destination already exists; use a fresh experiment directory")
    archive = subprocess.check_output(["git", "archive", "--format=tar", UPSTREAM_COMMIT], cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        members = tar.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"unsupported archive entry: {member.name}")
        dockerfile = tar.extractfile("Dockerfile").read()
        if hashlib.sha256(dockerfile).hexdigest() != DOCKERFILE_SHA:
            raise ValueError("upstream Dockerfile is not the reviewed source")
        destination.mkdir(parents=True)
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
                target.chmod(member.mode & 0o777)

    text = dockerfile.decode()
    text = replace_once(text, "ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04", f"ARG CUDA_IMAGE={CUDA_IMAGE}")
    text = replace_once(text, "\n    cd nccl && make -j ${BUILD_JOBS}",
                        f"\n    cd nccl && git checkout --detach {NCCL_COMMIT} && make -j ${{BUILD_JOBS}}")
    text = replace_once(text, "WORKDIR /workspace/flashinfer\n", """WORKDIR /workspace/flashinfer
ARG ROCE_SINGLE_BLOCK_ARM=control
COPY roce-single-block/ /tmp/roce-single-block/
RUN uv run --no-project --no-config --python /usr/bin/python3 /tmp/roce-single-block/apply_kernel.py "$ROCE_SINGLE_BLOCK_ARM"
""")
    text += """
# Record the actual treatment and retain its GPU regression sources.
COPY --from=flashinfer_wheels /.roce-single-block.json /workspace/roce-single-block.json
COPY --from=flashinfer_wheels /kernel-tests/ /opt/roce-single-block-tests/
ARG ROCE_SINGLE_BLOCK_ARM=control
LABEL glm53.roce-single-block.arm=$ROCE_SINGLE_BLOCK_ARM
"""
    (destination / "Dockerfile").write_text(text)
    bundle = destination / "roce-single-block"
    bundle.mkdir()
    source = Path(__file__).resolve().parent
    for name in ("apply_kernel.py", "kernel.patch", "manifest.json"):
        shutil.copyfile(source / name, bundle / name)
    shutil.copyfile(source / "build_pair.sh", destination / "build_pair.sh")
    # Reuse the frozen read-only both-node policy for build readiness, without
    # changing the production unit or installing the helper into its runtime.
    shutil.copyfile(source.parent / "startup-memory/wait_cluster_idle.py",
                    destination / "wait_cluster_idle.py")
    metadata = {"experiment": "roce-single-block", "upstream_launcher": UPSTREAM_COMMIT,
                "cuda_image": CUDA_IMAGE, "nccl_commit": NCCL_COMMIT,
                "vllm_commit": VLLM_COMMIT, "flashinfer_commit": FLASHINFER_COMMIT,
                "torch": "2.13.0+cu130", "cutlass_dsl": "4.7.0", "architecture": "12.1a",
                "gpu_qualified": False}
    # JSON is also valid YAML; the upstream Dockerfile retains this provenance.
    (destination / "build-metadata.yaml").write_text(json.dumps(metadata, indent=2) + "\n")
    (destination / ".dockerignore").write_text((destination / ".dockerignore").read_text() + "\nartifacts/\ncontext-sha256.txt\n")
    hashes = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            hashes.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(destination)}")
    (destination / "context-sha256.txt").write_text("\n".join(hashes) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.upstream_repo, args.out), indent=2))
