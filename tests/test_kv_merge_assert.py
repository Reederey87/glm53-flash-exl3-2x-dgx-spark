#!/usr/bin/env python3
"""CPU tests for the #55234 KVCacheSpec.merge assert overlay."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "overlay" / "patch_kv_merge_assert.py"
FIXTURE = ROOT / "tests/fixtures/source-compat/kv_cache_interface.py"


def apply_patch(tmp_path: Path) -> Path:
    target = tmp_path / "kv_cache_interface.py"
    shutil.copy2(FIXTURE, target)
    env = os.environ.copy()
    env["GLM53_KV_CACHE_INTERFACE_PY"] = str(target)
    subprocess.run([sys.executable, "-S", str(PATCH)], check=True, env=env)
    subprocess.run([sys.executable, "-S", str(PATCH)], check=True, env=env)
    return target


def load_module(path: Path, name: str = "patched_kv_cache_interface"):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_overlay_keeps_incompatibility_under_optimized_python(tmp_path: Path) -> None:
    target = apply_patch(tmp_path)
    source = target.read_text()
    assert "assert all(spec == specs[0] for spec in specs[1:])" not in source
    assert "raise AssertionError(  # [glm53-kv-merge-assert]" in source
    assert source.count("# [glm53-kv-merge-assert]") == 1

    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            (
                "import importlib.util, sys\n"
                f"path = {str(target)!r}\n"
                "spec = importlib.util.spec_from_file_location('kv', path)\n"
                "mod = importlib.util.module_from_spec(spec)\n"
                "sys.modules['kv'] = mod\n"
                "spec.loader.exec_module(mod)\n"
                "a = mod.KVCacheSpec(block_size=64, extra='a')\n"
                "b = mod.KVCacheSpec(block_size=64, extra='b')\n"
                "try:\n"
                "    mod.KVCacheSpec.merge([a, b])\n"
                "except AssertionError as exc:\n"
                "    print(type(exc).__name__ + ':' + str(exc))\n"
                "else:\n"
                "    raise SystemExit('merge swallowed the incompatibility')\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "AssertionError:All layers in the same KV cache group must be the same." in result.stdout


def test_unpatched_assert_vanishes_under_optimized_python(tmp_path: Path) -> None:
    target = tmp_path / "kv_cache_interface.py"
    shutil.copy2(FIXTURE, target)
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            (
                "import importlib.util, sys\n"
                f"path = {str(target)!r}\n"
                "spec = importlib.util.spec_from_file_location('kv', path)\n"
                "mod = importlib.util.module_from_spec(spec)\n"
                "sys.modules['kv'] = mod\n"
                "spec.loader.exec_module(mod)\n"
                "a = mod.KVCacheSpec(block_size=64, extra='a')\n"
                "b = mod.KVCacheSpec(block_size=64, extra='b')\n"
                "merged = mod.KVCacheSpec.merge([a, b])\n"
                "print(merged.extra)\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "a"


def test_overlay_is_idempotent_and_keeps_any_merge(tmp_path: Path) -> None:
    target = apply_patch(tmp_path)
    module = load_module(target)
    a = module.MLAAttentionSpec(block_size=64, extra="x", non_causal_multi_token_decode=False)
    b = module.MLAAttentionSpec(block_size=64, extra="y", non_causal_multi_token_decode=True)
    merged = module.MLAAttentionSpec.merge([a, b])
    assert merged.non_causal_multi_token_decode is True
    assert a.real_page_size_bytes == 64 * 584
    fp8 = module.MLAAttentionSpec(block_size=64, cache_dtype_str="fp8_ds_mla")
    assert fp8.real_page_size_bytes == 64 * 656


def test_overlay_rejects_drifted_installed_body_without_writing(tmp_path: Path) -> None:
    target = apply_patch(tmp_path)
    drifted = target.read_text().replace(
        'raise AssertionError(  # [glm53-kv-merge-assert]\n'
        '                "All layers in the same KV cache group must be the same."',
        'raise AssertionError(  # [glm53-kv-merge-assert]\n'
        '                "layers must match, maybe"',
        1,
    )
    assert drifted != target.read_text()
    target.write_text(drifted)
    before = target.read_bytes()
    env = os.environ.copy()
    env["GLM53_KV_CACHE_INTERFACE_PY"] = str(target)
    result = subprocess.run(
        [sys.executable, "-S", str(PATCH)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "kv-merge-assert preflight failed" in result.stderr
    assert target.read_bytes() == before


def test_overlay_rejects_return_block_drift_without_writing(tmp_path: Path) -> None:
    target = tmp_path / "kv_cache_interface.py"
    shutil.copy2(FIXTURE, target)
    target.write_text(
        target.read_text().replace("return copy.deepcopy(specs[0])", "return specs[0]", 1)
    )
    before = target.read_bytes()
    env = os.environ.copy()
    env["GLM53_KV_CACHE_INTERFACE_PY"] = str(target)
    result = subprocess.run(
        [sys.executable, "-S", str(PATCH)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "anchor=0" in result.stderr
    assert target.read_bytes() == before


def test_overlay_rejects_duplicate_anchors_without_writing(tmp_path: Path) -> None:
    target = tmp_path / "kv_cache_interface.py"
    shutil.copy2(FIXTURE, target)
    source = target.read_text()
    spec = importlib.util.spec_from_file_location("kv_merge_overlay", PATCH)
    assert spec and spec.loader
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    target.write_text(source.replace(overlay.ANCHOR, overlay.ANCHOR + overlay.ANCHOR, 1))
    before = target.read_bytes()
    env = os.environ.copy()
    env["GLM53_KV_CACHE_INTERFACE_PY"] = str(target)
    result = subprocess.run(
        [sys.executable, "-S", str(PATCH)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "anchor=2" in result.stderr
    assert target.read_bytes() == before


def test_overlay_rejects_missing_anchor_without_writing(tmp_path: Path) -> None:
    target = tmp_path / "kv_cache_interface.py"
    target.write_text("class KVCacheSpec:\n    pass\n")
    before = target.read_bytes()
    env = os.environ.copy()
    env["GLM53_KV_CACHE_INTERFACE_PY"] = str(target)
    result = subprocess.run(
        [sys.executable, "-S", str(PATCH)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "inherited KVCacheSpec.merge assert drifted" in result.stderr
    assert target.read_bytes() == before
