from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def guard(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "experiments/dense-int64"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("uv_cache_guard", folder / "release_uv_build_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def record(guard):
    return {"ID": guard.CACHE_ID, "Type": "exec.cachemount", "Reclaimable": True,
            "Shared": False, "Description": 'cached mount /root/.cache/uv from exec uv pip install with id "/uv-cache"'}


def mount(guard):
    return f"30 20 1:2 {guard.SOURCE} /completed-uv ro,relatime - ext4 /dev/nvme rw"


def test_exact_idle_record_and_readonly_mount(guard, record):
    guard.verify_cache(record, mount(guard))


@pytest.mark.parametrize("key,value", [
    ("ID", "another"), ("Type", "regular"), ("Reclaimable", False),
    ("Reclaimable", "true"), ("Shared", True), ("Description", "unrelated cache"),
])
def test_wrong_or_in_use_cache_rejected(guard, record, key, value):
    record[key] = value
    with pytest.raises(ValueError, match="inactive"):
        guard.verify_cache(record, mount(guard))


@pytest.mark.parametrize("kind", ["missing", "wrong", "writable", "duplicate", "nested"])
def test_unsafe_mount_rejected(guard, record, kind):
    row = mount(guard)
    rows = {"missing": "", "wrong": row.replace(guard.SOURCE, "/home/nvidia/.cache/huggingface"),
            "writable": row.replace("ro,relatime", "rw,relatime"),
            "duplicate": row + "\n" + row,
            "nested": row + "\n31 30 1:3 /outside /completed-uv/archive-v0 ro - ext4 /dev/nvme rw"}
    with pytest.raises(ValueError, match="read-only"):
        guard.verify_cache(record, rows[kind])


def test_observed_43_byte_uv_tag_and_required_layout(guard, tmp_path):
    # Exact bytes read from the real completed cache, also specified by CACHEDIR.TAG.
    tag = tmp_path / "CACHEDIR.TAG"
    tag.write_bytes(b"Signature: 8a477f597d28d172789f06886806bc55")
    for name in ("archive-v0", "wheels-v7"):
        (tmp_path / name).mkdir()
    guard.verify_layout(tmp_path)
    tag.write_bytes(b"wrong signature")
    with pytest.raises(ValueError, match="signature"):
        guard.verify_layout(tmp_path)
    tag.write_bytes(b"Signature: 8a477f597d28d172789f06886806bc55")
    (tmp_path / "wheels-v7").rmdir()
    (tmp_path / "wheels-v7").symlink_to(tmp_path / "archive-v0", target_is_directory=True)
    with pytest.raises(ValueError, match="layout"):
        guard.verify_layout(tmp_path)
