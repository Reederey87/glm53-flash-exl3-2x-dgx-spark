from __future__ import annotations

import importlib.util
from pathlib import Path
import stat

import pytest


@pytest.fixture
def caches(tmp_path):
    path = Path(__file__).resolve().parents[1] / "experiments/dense-int64/preserve_compile_caches.py"
    spec = importlib.util.spec_from_file_location("preserve_caches", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for relative in module.CACHES:
        folder = tmp_path / relative
        folder.mkdir(parents=True)
        folder.chmod(0o775)
        (folder / "artifact").write_text(relative)
    model = tmp_path / ".cache/huggingface/model"
    model.parent.mkdir()
    model.write_text("unchanged checkpoint")
    return module, tmp_path


def test_preserves_inodes_contents_permissions_and_checkpoint(caches):
    module, root = caches
    before = {name: (root / name).stat().st_ino for name in module.CACHES}
    result = module.preserve(root, "incumbent")
    assert len(result) == 5
    for row, name in zip(result, module.CACHES):
        fresh, saved = Path(row["cache"]), Path(row["preserved"])
        assert saved.stat().st_ino == before[name]
        assert (saved / "artifact").read_text() == name
        assert not list(fresh.iterdir())
        assert stat.S_IMODE(fresh.stat().st_mode) == 0o775
    assert (root / ".cache/huggingface/model").read_text() == "unchanged checkpoint"


@pytest.mark.parametrize("fault", ["missing", "file", "symlink", "parent_link", "backup", "backup_link", "arm"])
def test_refuses_before_any_rename(caches, fault):
    module, root = caches
    source = root / ".tilelang"
    if fault in ("missing", "file", "symlink"):
        (source / "artifact").unlink()
        source.rmdir()
        if fault == "file":
            source.write_text("not a cache directory")
        elif fault == "symlink":
            source.symlink_to(root / ".triton", target_is_directory=True)
    elif fault == "parent_link":
        (root / ".cache").rename(root / "actual-cache")
        (root / ".cache").symlink_to(root / "actual-cache", target_is_directory=True)
    elif fault in ("backup", "backup_link"):
        backup = source.with_name(source.name + ".int64-before-incumbent-20261103")
        if fault == "backup":
            backup.mkdir()
        else:
            backup.symlink_to(root / "missing-target")
    with pytest.raises(ValueError):
        module.preserve(root, "invalid" if fault == "arm" else "incumbent")
    assert (root / ".triton/artifact").read_text() == ".triton"
    assert not (root / ".triton.int64-before-incumbent-20261103").exists()


def test_partial_creation_failure_restores_all_originals(caches, monkeypatch):
    module, root = caches
    original_mkdir = Path.mkdir
    before = {name: (root / name).stat().st_ino for name in module.CACHES}

    def fail_second(path, *args, **kwargs):
        if path == root / ".cache/flashinfer":
            raise OSError("injected mkdir failure")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_second)
    with pytest.raises(OSError, match="injected mkdir failure"):
        module.preserve(root, "incumbent")
    for name in module.CACHES:
        assert (root / name).stat().st_ino == before[name]
        assert (root / name / "artifact").read_text() == name
        assert not (root / name).with_name(Path(name).name + ".int64-before-incumbent-20261103").exists()


def test_retry_retains_original_and_failed_start_separately(caches):
    module, root = caches
    original = module.preserve(root, 'incumbent')
    for name in module.CACHES:
        (root / name / 'failed-start').write_text('preserve this too')
    retry = module.preserve(root, 'incumbent-retry1')
    for old, new in zip(original, retry):
        assert (Path(old['preserved']) / 'artifact').exists()
        assert (Path(new['preserved']) / 'failed-start').read_text() == 'preserve this too'
        assert not list(Path(new['cache']).iterdir())
