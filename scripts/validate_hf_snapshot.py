#!/usr/bin/env python3
"""Validate one selected Hugging Face safetensors snapshot.

The Hub cache stores snapshot files as symlinks into ``../../blobs``. Counting
only regular files therefore reports zero for a healthy cache. This validator
instead reads the selected snapshot's index, follows valid symlinks, and checks
that every referenced safetensors file contains its complete declared payload.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class SnapshotError(ValueError):
    """The selected snapshot is absent, inconsistent, or incomplete."""


@dataclass(frozen=True)
class SnapshotReport:
    revision: str
    snapshot: str
    index: str
    shard_count: int
    total_bytes: int


def _read_ref(repo: Path) -> str:
    try:
        return (repo / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def select_revision(repo: Path, explicit_revision: str = "") -> str:
    """Select an explicit revision, refs/main, or finally the newest snapshot."""
    if explicit_revision:
        return explicit_revision
    ref = _read_ref(repo)
    if ref:
        return ref
    snapshots = repo / "snapshots"
    try:
        candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    except OSError as exc:
        raise SnapshotError(f"no snapshots under {repo}: {exc}") from exc
    if not candidates:
        raise SnapshotError(f"no snapshots under {repo}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns).name


def _safe_snapshot_member(snapshot: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise SnapshotError(f"unsafe shard path in index: {name!r}")
    return snapshot.joinpath(*relative.parts)


def _validate_safetensors_file(path: Path) -> int:
    if not path.exists():
        kind = "dangling symlink" if path.is_symlink() else "missing file"
        raise SnapshotError(f"{kind}: {path}")
    if not path.is_file():
        raise SnapshotError(f"not a regular file: {path}")
    size = path.stat().st_size
    if size < 9:
        raise SnapshotError(f"zero/truncated safetensors file ({size} bytes): {path}")
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        header_len = int.from_bytes(raw_len, "little", signed=False)
        if header_len <= 0 or header_len > size - 8:
            raise SnapshotError(
                f"invalid safetensors header length {header_len} for {size}-byte file: {path}"
            )
        raw_header = handle.read(header_len)
    try:
        header: Any = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"invalid safetensors JSON header: {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise SnapshotError(f"safetensors header is not an object: {path}")

    max_payload_end = 0
    tensor_count = 0
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        tensor_count += 1
        if not isinstance(metadata, dict):
            raise SnapshotError(f"invalid tensor metadata for {name!r}: {path}")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            raise SnapshotError(f"invalid data_offsets for {name!r}: {path}")
        max_payload_end = max(max_payload_end, offsets[1])
    if tensor_count == 0:
        raise SnapshotError(f"safetensors file declares no tensors: {path}")
    payload_bytes = size - 8 - header_len
    if max_payload_end > payload_bytes:
        raise SnapshotError(
            f"truncated safetensors payload ({payload_bytes} < {max_payload_end} bytes): {path}"
        )
    return size


def validate_snapshot(
    repo: Path, explicit_revision: str = "", expected_shards: int | None = None
) -> SnapshotReport:
    revision = select_revision(repo, explicit_revision)
    snapshot = repo / "snapshots" / revision
    if not snapshot.is_dir():
        source = "explicit revision" if explicit_revision else "selected revision"
        raise SnapshotError(f"{source} {revision} is missing under {repo / 'snapshots'}")

    index = snapshot / "model.safetensors.index.json"
    if not index.is_file():
        raise SnapshotError(f"safetensors index missing: {index}")
    try:
        index_data = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"invalid safetensors index {index}: {exc}") from exc
    weight_map = index_data.get("weight_map") if isinstance(index_data, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise SnapshotError(f"safetensors index has no non-empty weight_map: {index}")
    if not all(isinstance(name, str) and name for name in weight_map.values()):
        raise SnapshotError(f"safetensors index contains an invalid shard name: {index}")

    shard_names = sorted(set(weight_map.values()))
    if expected_shards is not None and len(shard_names) != expected_shards:
        raise SnapshotError(
            f"index references {len(shard_names)} shards, expected {expected_shards}: {index}"
        )

    indexed = {PurePosixPath(name).as_posix() for name in shard_names}
    present = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*.safetensors")
    }
    unindexed = sorted(present - indexed)
    if unindexed:
        sample = ", ".join(unindexed[:3])
        raise SnapshotError(f"safetensors files missing from index: {sample}")

    total_bytes = 0
    for name in shard_names:
        total_bytes += _validate_safetensors_file(_safe_snapshot_member(snapshot, name))
    return SnapshotReport(
        revision=revision,
        snapshot=str(snapshot),
        index=str(index),
        shard_count=len(shard_names),
        total_bytes=total_bytes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument(
        "--field",
        choices=("count", "revision", "snapshot", "json"),
        default="json",
    )
    args = parser.parse_args()
    try:
        report = validate_snapshot(args.repo, args.revision, args.expected_shards)
    except SnapshotError as exc:
        print(f"snapshot validation failed: {exc}", file=sys.stderr)
        return 1
    if args.field == "count":
        print(report.shard_count)
    elif args.field == "revision":
        print(report.revision)
    elif args.field == "snapshot":
        print(report.snapshot)
    else:
        print(json.dumps(asdict(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
