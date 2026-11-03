from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def export(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "experiments/dense-int64"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("export_cache", folder / "release_vllm_export_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fault", [None, "writable", "wrong source", "nested", "duplicate", "missing"])
def test_exact_export_mount_only(export, fault):
    line = f"2 1 0:2 {export.SOURCE} {export.ROOT} ro - ext4 /dev/nvme rw"
    if fault == "writable":
        line = line.replace(" ro -", " rw -")
    elif fault == "wrong source":
        line = line.replace(export.SOURCE, "/other/export")
    elif fault == "nested":
        line += f"\n3 2 0:3 /other {export.ROOT}/nested ro - ext4 /dev/nvme rw"
    elif fault == "duplicate":
        line += "\n" + line
    elif fault == "missing":
        line = ""
    if fault is None:
        export.verify_mount(line)
    else:
        with pytest.raises(ValueError):
            export.verify_mount(line)


@pytest.mark.parametrize("fault", [None, "hash", "size", "symlink"])
def test_hash_before_advice_and_file_preserved(export, tmp_path, monkeypatch, fault):
    data = b"verified wheel bytes"
    path = tmp_path / "wheel.whl"
    path.write_bytes(data)
    monkeypatch.setattr(export, "SHA", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(export, "SIZE", len(data))
    calls = []
    monkeypatch.setattr(export.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(export.os, "posix_fadvise", lambda *args: calls.append(args), raising=False)
    if fault == "hash":
        path.write_bytes(b"x" * len(data))
    elif fault == "size":
        path.write_bytes(data + b"x")
    elif fault == "symlink":
        link = tmp_path / "link.whl"
        link.symlink_to(path)
        path = link
    before = path.read_bytes()
    if fault is None:
        report = export.release(path)
        assert report["metadata_unchanged"]
        assert len(calls) == 1
        assert calls[0][1:] == (0, 0, 4)
    else:
        with pytest.raises((ValueError, OSError)):
            export.release(path)
        assert not calls
    assert path.read_bytes() == before
