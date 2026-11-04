from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def caches(tmp_path):
    path = Path(__file__).resolve().parents[1] / "experiments/dense-int64/restore_compile_caches.py"
    spec = importlib.util.spec_from_file_location("restore_caches", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = {"ok": True, "arm": "incumbent", "caches": []}
    for relative in module.CACHES:
        active = tmp_path / relative
        active.mkdir(parents=True)
        active.chmod(0o775)
        (active / "arm-artifact").write_text(relative)
        saved = active.with_name(active.name + ".int64-before-incumbent-20261103")
        saved.mkdir()
        saved.chmod(0o775)
        (saved / "original-artifact").write_text(relative)
        identity = module.identity(saved)
        identity.pop("mtime_ns")
        original["caches"].append({"cache": str(active), "preserved": str(saved), "fresh_empty": True, **identity})
    checkpoint = tmp_path / ".cache/huggingface/checkpoint"
    checkpoint.parent.mkdir()
    checkpoint.write_text("unchanged")
    return module, tmp_path, original


def test_readonly_plan_and_restore_preserve_every_identity(caches):
    module, root, original = caches
    rows = module.plan(root, original, "candidate")
    assert all((Path(row["active"]) / "arm-artifact").exists() for row in rows)
    assert all(not Path(row["retain_final_arm"]).exists() for row in rows)
    module.restore(rows)
    for row in rows:
        active, archive = Path(row["active"]), Path(row["retain_final_arm"])
        assert module.identity(active) == row["original_identity"]
        assert module.identity(archive) == row["active_identity"]
        assert (active / "original-artifact").read_text() == str(active.relative_to(root))
        assert (archive / "arm-artifact").exists()
        assert not Path(row["original"]).exists()
    assert (root / ".cache/huggingface/checkpoint").read_text() == "unchanged"


@pytest.mark.parametrize("fault", ["archive", "archive_link", "source_link", "original_link", "wrong_inode", "missing_row", "duplicate_row", "wrong_arm", "wrong_path", "wrong_owner"])
def test_plan_refuses_before_any_rename(caches, fault, monkeypatch):
    module, root, original = caches
    first = root / module.CACHES[0]
    saved = Path(original["caches"][0]["preserved"])
    archive = first.with_name(first.name + ".int64-after-candidate-20261103")
    if fault == "archive":
        archive.mkdir()
    elif fault == "archive_link":
        archive.symlink_to(root / "absent")
    elif fault == "source_link":
        first.rename(root / "actual")
        first.symlink_to(root / "actual", target_is_directory=True)
    elif fault == "original_link":
        saved.rename(root / "saved")
        saved.symlink_to(root / "saved", target_is_directory=True)
    elif fault == "wrong_inode":
        original["caches"][0]["inode"] += 1
    elif fault == "missing_row":
        original["caches"].pop()
    elif fault == "duplicate_row":
        original["caches"][1] = deepcopy(original["caches"][0])
    elif fault == "wrong_arm":
        original["arm"] = "incumbent-retry1"
    elif fault == "wrong_path":
        original["caches"][0]["preserved"] = str(root / "wrong")
    else:
        monkeypatch.setattr(module.os, "geteuid", lambda: -1)
    with pytest.raises(ValueError):
        module.plan(root, original, "candidate")
    assert (root / ".tilelang/arm-artifact").exists()
    assert (root / ".tilelang.int64-before-incumbent-20261103/original-artifact").exists()


@pytest.mark.parametrize("fail_call", [1, 2, 3, 6, 10])
def test_partial_failure_reverses_all_completed_renames(caches, monkeypatch, fail_call):
    module, root, original = caches
    rows = module.plan(root, original, "candidate")
    rename = Path.rename
    calls = 0

    def fail_once(source, target):
        nonlocal calls
        calls += 1
        if calls == fail_call:
            raise OSError("injected rename failure")
        return rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_once)
    with pytest.raises(OSError, match="injected"):
        module.restore(rows)
    for row in rows:
        assert module.identity(Path(row["active"])) == row["active_identity"]
        assert module.identity(Path(row["original"])) == row["original_identity"]
        assert not Path(row["retain_final_arm"]).exists()


def test_stale_plan_refuses_before_first_rename(caches):
    module, root, original = caches
    rows = module.plan(root, original, "candidate")
    (root / ".tilelang/new-artifact").write_text("unexpected writer")
    with pytest.raises(ValueError, match="changed"):
        module.restore(rows)
    assert all(Path(row["original"]).exists() and not Path(row["retain_final_arm"]).exists() for row in rows)


def test_exact_original_receipt_hash_is_required(caches, monkeypatch):
    module, root, original = caches
    path = root / "original.json"
    raw = json.dumps(original).encode()
    path.write_bytes(raw)
    monkeypatch.setitem(module.ORIGINAL_HASHES, "head", hashlib.sha256(raw).hexdigest())
    assert module.original_receipt(path, "head") == original
    path.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="hash"):
        module.original_receipt(path, "head")


def test_live_container_prevents_apply(caches, monkeypatch):
    module, _, _ = caches
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "live-container\n")
    with pytest.raises(ValueError, match="containers"):
        module.local_idle_guard("head")


def test_rollback_never_overwrites_an_intervening_writer(caches, monkeypatch):
    module, root, original = caches
    rows = module.plan(root, original, "candidate")
    rename = Path.rename
    active = Path(rows[0]["active"])
    saved = Path(rows[0]["original"])

    def interfere(source, target):
        if source == saved:
            active.mkdir()
            (active / "unexpected-writer").write_text("preserve")
            raise OSError("intervening writer")
        return rename(source, target)

    monkeypatch.setattr(Path, "rename", interfere)
    with pytest.raises(ValueError, match="rollback destination occupied"):
        module.restore(rows)
    assert (active / "unexpected-writer").read_text() == "preserve"
    assert (saved / "original-artifact").exists()
    assert (Path(rows[0]["retain_final_arm"]) / "arm-artifact").exists()
    assert all(Path(row["original"]).exists() for row in rows[1:])
