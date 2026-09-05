from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_hf_snapshot.py"
START = ROOT / "start.sh"


def _module():
    spec = importlib.util.spec_from_file_location("validate_hf_snapshot", VALIDATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _shell_function(name: str) -> str:
    source = START.read_text(encoding="utf-8")
    match = re.search(rf"(?ms)^{re.escape(name)}\(\) \{{\n.*?^\}}\n", source)
    assert match, f"missing function {name}"
    return match.group(0)


def _safetensors(path: Path, payload_bytes: int = 8, truncate: int = 0) -> None:
    header = json.dumps(
        {"tensor": {"dtype": "U8", "shape": [payload_bytes], "data_offsets": [0, payload_bytes]}}
    ).encode()
    raw = len(header).to_bytes(8, "little") + header + (b"x" * payload_bytes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw[:-truncate] if truncate else raw)


def _snapshot(repo: Path, revision: str, *, symlinks: bool = False) -> Path:
    snap = repo / "snapshots" / revision
    snap.mkdir(parents=True)
    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    weight_map = {"a": names[0], "b": names[1]}
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (snap / "config.json").write_text("{}")
    if symlinks:
        blobs = repo / "blobs"
        blobs.mkdir(parents=True)
        for index, name in enumerate(names):
            blob = blobs / f"blob-{index}"
            _safetensors(blob)
            (snap / name).symlink_to(os.path.relpath(blob, snap))
    else:
        for name in names:
            _safetensors(snap / name)
    return snap


def test_regular_and_hub_symlink_snapshots_validate(tmp_path: Path) -> None:
    module = _module()
    regular = tmp_path / "regular"
    linked = tmp_path / "linked"
    _snapshot(regular, "regular-rev")
    _snapshot(linked, "linked-rev", symlinks=True)
    assert module.validate_snapshot(regular, "regular-rev", 2).shard_count == 2
    assert module.validate_snapshot(linked, "linked-rev", 2).shard_count == 2


def test_explicit_revision_wins_over_stale_refs_main(tmp_path: Path) -> None:
    module = _module()
    _snapshot(tmp_path, "wanted")
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "main").write_text("missing")
    report = module.validate_snapshot(tmp_path, "wanted", 2)
    assert report.revision == "wanted"
    assert module.select_revision(tmp_path, "wanted") == "wanted"


def test_dangling_missing_unindexed_and_truncated_shards_fail(tmp_path: Path) -> None:
    module = _module()

    dangling = tmp_path / "dangling"
    snap = _snapshot(dangling, "rev", symlinks=True)
    (dangling / "blobs" / "blob-1").unlink()
    try:
        module.validate_snapshot(dangling, "rev", 2)
    except module.SnapshotError as exc:
        assert "dangling symlink" in str(exc)
    else:
        raise AssertionError("dangling symlink accepted")

    missing = tmp_path / "missing"
    snap = _snapshot(missing, "rev")
    (snap / "model-00002-of-00002.safetensors").unlink()
    try:
        module.validate_snapshot(missing, "rev", 2)
    except module.SnapshotError as exc:
        assert "missing file" in str(exc)
    else:
        raise AssertionError("missing shard accepted")

    unindexed = tmp_path / "unindexed"
    snap = _snapshot(unindexed, "rev")
    _safetensors(snap / "model-00003-of-00003.safetensors")
    try:
        module.validate_snapshot(unindexed, "rev", 2)
    except module.SnapshotError as exc:
        assert "missing from index" in str(exc)
    else:
        raise AssertionError("unindexed shard accepted")

    truncated = tmp_path / "truncated"
    snap = _snapshot(truncated, "rev")
    _safetensors(snap / "model-00002-of-00002.safetensors", truncate=4)
    try:
        module.validate_snapshot(truncated, "rev", 2)
    except module.SnapshotError as exc:
        assert "truncated safetensors payload" in str(exc)
    else:
        raise AssertionError("truncated shard accepted")


def test_zero_file_and_incomplete_index_fail(tmp_path: Path) -> None:
    module = _module()
    zero = tmp_path / "zero"
    snap = _snapshot(zero, "rev")
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"")
    try:
        module.validate_snapshot(zero, "rev", 2)
    except module.SnapshotError as exc:
        assert "zero/truncated" in str(exc)
    else:
        raise AssertionError("zero shard accepted")

    incomplete = tmp_path / "incomplete"
    snap = _snapshot(incomplete, "rev")
    index = snap / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}}))
    try:
        module.validate_snapshot(incomplete, "rev", 2)
    except module.SnapshotError as exc:
        assert "references 1 shards, expected 2" in str(exc)
    else:
        raise AssertionError("incomplete index accepted")


def test_cli_count_and_start_wiring(tmp_path: Path) -> None:
    _snapshot(tmp_path, "wanted", symlinks=True)
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--repo",
            str(tmp_path),
            "--revision",
            "wanted",
            "--expected-shards",
            "2",
            "--field",
            "count",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2"
    source = START.read_text(encoding="utf-8")
    assert 'MODEL_SELECTED_REVISION="${MODEL_REVISION:-}"' in source
    assert "MODEL_REVISION_EXPLICIT=1" in source
    assert 'count_shards "$MODEL_PATH" "$MODEL_SELECTED_REVISION"' in source
    assert 'sync_repo_to_worker "$MODEL_PATH" "$MODEL_CACHE_NAME" "weights" "$MODEL_SELECTED_REVISION"' in source
    assert '_glm53_model_revision_set="${MODEL_REVISION+x}"' in source
    assert 'elif [ "$MODEL" = "Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw" ]' in source


def test_explicit_revision_cannot_fall_through_to_fallback(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    fallback = tmp_path / "fallback"
    _snapshot(fallback, "fallback-main")
    (fallback / "refs").mkdir()
    (fallback / "refs" / "main").write_text("fallback-main")
    script = f"""
set -euo pipefail
SCRIPT_DIR={shlex.quote(str(ROOT))}
EXPECTED_SHARDS=2
MODEL_PATH={shlex.quote(str(primary))}
FALLBACK_MODEL_PATH={shlex.quote(str(fallback))}
MODEL_CACHE_NAME=primary
MODEL_FALLBACK_CACHE_NAME=fallback
MODEL_FALLBACK=example/fallback
MODEL_SELECTED_REVISION=wrong-explicit-pin
MODEL_REVISION_EXPLICIT=1
log() {{ :; }}
{_shell_function("ensure_refs_main")}
{_shell_function("count_shards")}
{_shell_function("adopt_complete_weights")}
if adopt_complete_weights; then
    echo "unexpected-success:$MODEL_PATH:$MODEL_SELECTED_REVISION"
    exit 9
fi
printf '%s|%s\\n' "$MODEL_PATH" "$MODEL_SELECTED_REVISION"
"""
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{primary}|wrong-explicit-pin"


def test_implicit_default_may_adopt_valid_fallback(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    fallback = tmp_path / "fallback"
    _snapshot(fallback, "fallback-main")
    (fallback / "refs").mkdir()
    (fallback / "refs" / "main").write_text("fallback-main")
    script = f"""
set -euo pipefail
SCRIPT_DIR={shlex.quote(str(ROOT))}
EXPECTED_SHARDS=2
MODEL_PATH={shlex.quote(str(primary))}
FALLBACK_MODEL_PATH={shlex.quote(str(fallback))}
MODEL_CACHE_NAME=primary
MODEL_FALLBACK_CACHE_NAME=fallback
MODEL_FALLBACK=example/fallback
MODEL_SELECTED_REVISION=implicit-default-pin
MODEL_REVISION_EXPLICIT=0
log() {{ :; }}
{_shell_function("ensure_refs_main")}
{_shell_function("count_shards")}
{_shell_function("adopt_complete_weights")}
adopt_complete_weights
printf '%s|%s|%s\\n' "$MODEL_PATH" "$MODEL_CACHE_NAME" "$MODEL_SELECTED_REVISION"
"""
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{fallback}|fallback|"
