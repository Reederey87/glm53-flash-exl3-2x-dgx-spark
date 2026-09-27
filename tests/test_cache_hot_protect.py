"""Installer for the reused-prefix mark. The rank tests live next to tail-first."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "overlay" / "patch_cache_hot_protect.py"

MANAGER = """\
        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
"""
UTILS = '''\
    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        self._block_hash = None
        self._block_hash_num_tokens = None
'''


def _run(tmp: Path, flag: str, pool_text: str) -> subprocess.CompletedProcess[str]:
    pool = tmp / "block_pool.py"
    manager = tmp / "manager.py"
    utils = tmp / "utils.py"
    pool.write_text(pool_text)
    manager.write_text(MANAGER)
    utils.write_text(UTILS)
    env = os.environ.copy()
    env["GLM53_CACHE_HOT_PROTECT"] = flag
    env["GLM53_BLOCK_POOL_PY"] = str(pool)
    env["GLM53_KV_MANAGER_PY"] = str(manager)
    env["GLM53_KV_UTILS_PY"] = str(utils)
    return subprocess.run(
        [sys.executable, str(INSTALLER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_flag_off_does_not_read_missing_files(tmp_path: Path):
    env = os.environ.copy()
    env["GLM53_CACHE_HOT_PROTECT"] = "0"
    env["GLM53_BLOCK_POOL_PY"] = str(tmp_path / "missing-pool.py")
    env["GLM53_KV_MANAGER_PY"] = str(tmp_path / "missing-manager.py")
    env["GLM53_KV_UTILS_PY"] = str(tmp_path / "missing-utils.py")
    result = subprocess.run(
        [sys.executable, str(INSTALLER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "unchanged" in result.stdout
    assert not (tmp_path / "missing-pool.py").exists()


def test_flag_on_requires_the_tail_marker(tmp_path: Path):
    result = _run(tmp_path, "1", "no marker here\n")
    assert result.returncode == 1
    assert (tmp_path / "manager.py").read_text() == MANAGER
    assert (tmp_path / "utils.py").read_text() == UTILS


def test_flag_on_patches_once(tmp_path: Path):
    result = _run(tmp_path, "1", "# [glm53-cache-tail-evict]\n")
    assert result.returncode == 0, result.stderr
    manager = (tmp_path / "manager.py").read_text()
    utils = (tmp_path / "utils.py").read_text()
    assert manager.count("[glm53-cache-hot-protect]") == 1
    assert "mark_reused" in manager
    assert utils.count("[glm53-cache-hot-protect]") == 1
    assert "clear_reused" in utils
    env = os.environ.copy()
    env["GLM53_CACHE_HOT_PROTECT"] = "1"
    env["GLM53_BLOCK_POOL_PY"] = str(tmp_path / "block_pool.py")
    env["GLM53_KV_MANAGER_PY"] = str(tmp_path / "manager.py")
    env["GLM53_KV_UTILS_PY"] = str(tmp_path / "utils.py")
    (tmp_path / "block_pool.py").write_text("# [glm53-cache-tail-evict]\n")
    again = subprocess.run(
        [sys.executable, str(INSTALLER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert again.returncode == 0, again.stderr
    assert (tmp_path / "manager.py").read_text() == manager
    assert (tmp_path / "utils.py").read_text() == utils


def test_drift_writes_nothing(tmp_path: Path):
    pool = tmp_path / "block_pool.py"
    manager = tmp_path / "manager.py"
    utils = tmp_path / "utils.py"
    pool.write_text("# [glm53-cache-tail-evict]\n")
    manager.write_text(MANAGER.replace("touch(new_computed_blocks)", "touch(blocks)"))
    utils.write_text(UTILS)
    before_m = manager.read_text()
    before_u = utils.read_text()
    env = os.environ.copy()
    env["GLM53_CACHE_HOT_PROTECT"] = "1"
    env["GLM53_BLOCK_POOL_PY"] = str(pool)
    env["GLM53_KV_MANAGER_PY"] = str(manager)
    env["GLM53_KV_UTILS_PY"] = str(utils)
    result = subprocess.run(
        [sys.executable, str(INSTALLER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert manager.read_text() == before_m
    assert utils.read_text() == before_u
