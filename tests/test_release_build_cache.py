from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


@pytest.fixture
def release_module(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "experiments/dense-int64/release_build_cache.py"
    spec = importlib.util.spec_from_file_location("release_build_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    return module


def test_advice_does_not_follow_links_read_payloads_or_duplicate_hardlinks(release_module, tmp_path, monkeypatch):
    root = tmp_path / "build"
    root.mkdir()
    artifact = root / "object.o"
    artifact.write_bytes(b"compiled bytes")
    os.link(artifact, root / "hardlink.o")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model").write_bytes(b"untouched")
    (root / "dir-link").symlink_to(outside, target_is_directory=True)
    (root / "file-link").symlink_to(outside / "model")
    calls = []
    monkeypatch.setattr(release_module.os, "posix_fadvise",
                        lambda fd, offset, length, advice: calls.append((os.fstat(fd).st_ino, offset, length, advice)), raising=False)
    before = artifact.stat()
    result = release_module.release_tree(root)
    assert calls == [(before.st_ino, 0, 0, 4)]
    assert result["files_advised"] == 1
    assert result["file_bytes_advised"] == len(b"compiled bytes")
    assert artifact.read_bytes() == b"compiled bytes"
    assert (outside / "model").read_bytes() == b"untouched"
    assert artifact.stat().st_mtime_ns == before.st_mtime_ns


def test_rejects_symlink_root_before_advice(release_module, tmp_path, monkeypatch):
    root = tmp_path / "link"
    root.symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setattr(release_module.os, "posix_fadvise", lambda *args: pytest.fail("unexpected advice"), raising=False)
    with pytest.raises(ValueError, match="real directory"):
        release_module.release_tree(root)


@pytest.mark.parametrize("workspace,tail", [("vllm", "vllm/vllm"), ("flashinfer", "flashinfer")])
def test_accepts_only_exact_completed_readonly_mount(release_module, workspace, tail):
    row = f"30 20 1:2 /var/lib/docker/overlay2/layer123/diff/workspace/{tail} /completed-{workspace} ro,relatime - ext4 /dev/nvme rw"
    assert release_module.completed_mount(workspace, row) == Path(f"/completed-{workspace}")


@pytest.mark.parametrize("source,mode,duplicate", [
    ("/home/nvidia/.cache/huggingface", "ro", False),
    ("/var/lib/docker/overlay2/layer/diff/workspace/flashinfer", "rw", False),
    ("/var/lib/docker/overlay2/layer/diff/workspace/flashinfer-extra", "ro", False),
    ("/var/lib/docker/overlay2/layer/diff/workspace/flashinfer/build", "ro", False),
    ("/var/lib/docker/overlay2/layer/diff/workspace/vllm/vllm", "ro", False),
    ("/var/lib/docker/overlay2/layer/../layer/diff/workspace/flashinfer", "ro", False),
    ("/var/lib/docker/overlay2/layer/diff/workspace/flashinfer", "ro", True),
])
def test_rejects_wrong_or_ambiguous_mount(release_module, source, mode, duplicate):
    row = f"30 20 1:2 {source} /completed-flashinfer {mode} - ext4 /dev/nvme rw"
    with pytest.raises(ValueError, match="mounted read-only"):
        release_module.completed_mount("flashinfer", row + ("\n" + row if duplicate else ""))


def test_rejects_unknown_workspace_and_missing_mount(release_module):
    with pytest.raises(ValueError, match="unknown"):
        release_module.completed_mount("huggingface", "")
    with pytest.raises(ValueError, match="mounted read-only"):
        release_module.completed_mount("flashinfer", "malformed mount row")
